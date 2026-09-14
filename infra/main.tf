locals {
  name_prefix = "${var.project}-${var.environment}"
  tags = {
    project     = var.project
    environment = var.environment
  }
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name_prefix}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_storage_account" "main" {
  name                            = replace("st${local.name_prefix}", "-", "")
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false
  min_tls_version                 = "TLS1_2"
  tags                            = local.tags

  blob_properties {
    delete_retention_policy {
      days = 7
    }
  }

  # Below are enterprise-hardening controls (private endpoints, CMK
  # encryption, SAS expiration policy, diagnostic logging, disabling shared
  # key auth, geo-redundant replication) that checkov flags but are
  # deliberately out of scope for this project's size and $190 budget --
  # they add cost/complexity disproportionate to a personal training run.
  # checkov:skip=CKV2_AZURE_41:no SAS tokens are issued
  # checkov:skip=CKV2_AZURE_33:no private endpoint for a project this size
  # checkov:skip=CKV2_AZURE_1:CMK adds Key Vault cost/complexity not justified here
  # checkov:skip=CKV_AZURE_33:queue service is unused, nothing to log
  # checkov:skip=CKV_AZURE_59:public access needed -- no VPN/private link set up
  # checkov:skip=CKV_AZURE_206:LRS is intentional; GRS is not justified for this budget
  # checkov:skip=CKV2_AZURE_40:Terraform's azurerm backend needs shared key auth here
}

resource "azurerm_storage_container" "data" {
  # checkov:skip=CKV2_AZURE_21:blob read logging not justified for a personal project
  name                  = "data"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "checkpoints" {
  # checkov:skip=CKV2_AZURE_21:blob read logging not justified for a personal project
  name                  = "checkpoints"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_container_registry" "main" {
  # Zone redundancy, geo-replication, quarantine/vulnerability scanning,
  # signed images, and private networking are Premium-tier ACR features --
  # disproportionate cost for a single-region, single-developer project.
  # checkov:skip=CKV_AZURE_237:dedicated data endpoints require Premium SKU
  # checkov:skip=CKV_AZURE_233:zone redundancy requires Premium SKU
  # checkov:skip=CKV_AZURE_167:untagged-manifest retention requires Premium SKU
  # checkov:skip=CKV_AZURE_164:content trust requires Premium SKU
  # checkov:skip=CKV_AZURE_165:single region deployment, no geo-replication needed
  # checkov:skip=CKV_AZURE_166:image quarantine requires Premium SKU
  # checkov:skip=CKV_AZURE_163:vulnerability scanning requires Premium SKU (Defender for Cloud)
  # checkov:skip=CKV_AZURE_139:disabling public network access requires Premium SKU
  name                = replace("acr${local.name_prefix}", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = local.tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.tags
}

resource "azurerm_key_vault" "main" {
  # purge_protection stays off deliberately: Phase 4 of this project ends
  # with a full `terraform destroy` when the Azure credits expire, and a
  # protected vault would soft-delete and block a clean re-apply under the
  # same name for its retention window.
  # checkov:skip=CKV_AZURE_42:purge protection would block the planned full teardown
  # checkov:skip=CKV_AZURE_189:public access needed -- no VPN/private link set up
  # checkov:skip=CKV_AZURE_109:no firewall rules justified for this project's size
  # checkov:skip=CKV_AZURE_110:see purge_protection comment above
  # checkov:skip=CKV2_AZURE_32:no private endpoint for a project this size
  name                     = "kv-${substr(local.name_prefix, 0, 18)}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "standard"
  purge_protection_enabled = false
  tags                     = local.tags
}

data "azurerm_client_config" "current" {}

resource "azurerm_consumption_budget_subscription" "main" {
  name            = "budget-${local.name_prefix}"
  subscription_id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}"

  amount     = var.monthly_budget_usd
  time_grain = "Monthly"

  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", timestamp())
  }

  dynamic "notification" {
    for_each = [50, 80, 95]
    content {
      enabled        = true
      threshold      = notification.value
      operator       = "GreaterThan"
      contact_emails = var.budget_alert_emails
    }
  }

  lifecycle {
    ignore_changes = [time_period]
  }
}

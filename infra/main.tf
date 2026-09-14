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
  name                     = replace("st${local.name_prefix}", "-", "")
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = local.tags
}

resource "azurerm_storage_container" "data" {
  name                  = "data"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "checkpoints" {
  name                  = "checkpoints"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_container_registry" "main" {
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

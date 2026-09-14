# Run once, locally, with local state, before infra/ can use the azurerm
# backend. terraform init && terraform apply in this directory only.
terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.15"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "location" {
  type    = string
  default = "westus3"
}

resource "azurerm_resource_group" "state" {
  name     = "rg-minigpt-tfstate"
  location = var.location
}

resource "azurerm_storage_account" "state" {
  name                            = "stminigpttfstate"
  resource_group_name             = azurerm_resource_group.state.name
  location                        = azurerm_resource_group.state.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false
  min_tls_version                 = "TLS1_2"

  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 7
    }
  }

  # See infra/main.tf for why these enterprise-hardening checks are
  # deliberately skipped for a project this size.
  # checkov:skip=CKV2_AZURE_41:no SAS tokens are issued
  # checkov:skip=CKV2_AZURE_33:no private endpoint for a project this size
  # checkov:skip=CKV2_AZURE_1:CMK adds Key Vault cost/complexity not justified here
  # checkov:skip=CKV_AZURE_33:queue service is unused, nothing to log
  # checkov:skip=CKV_AZURE_59:public access needed -- no VPN/private link set up
  # checkov:skip=CKV_AZURE_206:LRS is intentional; GRS is not justified for this budget
  # checkov:skip=CKV2_AZURE_40:Terraform's azurerm backend needs shared key auth here
}

resource "azurerm_storage_container" "tfstate" {
  # checkov:skip=CKV2_AZURE_21:blob read logging not justified for a personal project
  name                  = "tfstate"
  storage_account_id    = azurerm_storage_account.state.id
  container_access_type = "private"
}

output "backend_config" {
  value = {
    resource_group_name  = azurerm_resource_group.state.name
    storage_account_name = azurerm_storage_account.state.name
    container_name       = azurerm_storage_container.tfstate.name
    key                  = "minigpt.tfstate"
  }
}

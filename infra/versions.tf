terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.15"
    }
  }
  backend "azurerm" {
    # Values supplied via -backend-config in CI / infra/bootstrap output.
    # resource_group_name, storage_account_name, container_name = "tfstate", key = "minigpt.tfstate"
    use_azuread_auth = true
  }
}

provider "azurerm" {
  features {}
}

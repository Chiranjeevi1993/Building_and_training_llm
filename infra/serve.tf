resource "azurerm_user_assigned_identity" "container_app" {
  name                = "id-${local.name_prefix}-app"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_role_assignment" "app_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.container_app.principal_id
}

resource "azurerm_role_assignment" "app_storage_read" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.container_app.principal_id
}

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${local.name_prefix}"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

resource "azurerm_container_app" "serve" {
  name                         = "ca-${local.name_prefix}-serve"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_app.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_app.id
  }

  template {
    min_replicas = var.container_app_min_replicas
    max_replicas = 2

    container {
      name = "serve"
      # Placeholder tag; deploy.yml updates this to the built image digest.
      image  = "${azurerm_container_registry.main.login_server}/minigpt-serve:latest"
      cpu    = 2.0
      memory = "4Gi"

      env {
        name  = "MINIGPT_CONFIG"
        value = "configs/gpt2_small_tinystories.yaml"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}

output "container_app_fqdn" {
  value = azurerm_container_app.serve.ingress[0].fqdn
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "ml_workspace_name" {
  value = azurerm_machine_learning_workspace.main.name
}

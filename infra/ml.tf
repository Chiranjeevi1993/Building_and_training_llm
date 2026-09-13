resource "azurerm_machine_learning_workspace" "main" {
  name                    = "mlw-${local.name_prefix}"
  resource_group_name     = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id
  container_registry_id   = azurerm_container_registry.main.id
  tags                    = local.tags

  identity {
    type = "SystemAssigned"
  }
}

# Cluster map: one place to add/remove GPU tiers. Each is min_node_count = 0
# with a short idle scale-down so a forgotten cluster costs nothing.
locals {
  gpu_clusters = merge(
    {
      "gpu-a100" = {
        vm_size        = "Standard_NC24ads_A100_v4" # 1x A100 80GB
        max_nodes      = 2
        vm_priority    = "Dedicated"
        min_node_count = 0
      }
    },
    var.enable_a100 ? {
      "gpu-a100-x4" = {
        vm_size        = "Standard_NC96ads_A100_v4" # 4x A100 80GB — main training cluster
        max_nodes      = 1
        vm_priority    = "Dedicated"
        min_node_count = 0
      }
    } : {},
    var.enable_t4 ? {
      "gpu-t4-x4" = {
        vm_size        = "Standard_NC64as_T4_v3" # 4x T4 16GB — fallback if A100 quota refused
        max_nodes      = 1
        vm_priority    = "Dedicated"
        min_node_count = 0
      }
    } : {}
  )
}

resource "azurerm_machine_learning_compute_cluster" "gpu" {
  for_each                      = local.gpu_clusters
  name                          = each.key
  location                      = azurerm_resource_group.main.location
  vm_priority                   = each.value.vm_priority
  vm_size                       = each.value.vm_size
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id

  scale_settings {
    min_node_count                       = each.value.min_node_count
    max_node_count                       = each.value.max_nodes
    scale_down_nodes_after_idle_duration = "PT10M"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

resource "azurerm_role_assignment" "aml_storage" {
  for_each             = local.gpu_clusters
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.gpu[each.key].identity[0].principal_id
}

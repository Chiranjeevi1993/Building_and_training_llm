resource "azurerm_machine_learning_workspace" "main" {
  # checkov:skip=CKV2_AZURE_50:public network access is required here -- GitHub-hosted
  # runners submit jobs to this workspace and a private endpoint/VNet is disproportionate
  # cost/complexity for this project's size.
  # checkov:skip=CKV_AZURE_144:same reasoning -- public access is required, no VNet here
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
#
# NOTE: Azure ML validates vCPU quota against a cluster's *max* node count at
# creation time even when min_node_count = 0 (confirmed by an actual apply --
# ClusterMinNodesExceedCoreQuota). So every GPU cluster here is gated behind a
# flag, none apply unconditionally, and every flag must stay false until the
# matching quota is granted (see .plans/azure-migration.md Phase 0).
locals {
  gpu_clusters = merge(
    var.enable_a100 ? {
      "gpu-a100" = {
        vm_size        = "Standard_NC24ads_A100_v4" # 1x A100 80GB — smoke/sweep tier
        max_nodes      = 2
        vm_priority    = "Dedicated"
        min_node_count = 0
      }
    } : {},
    var.enable_a100_x4 ? {
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
  # checkov:skip=CKV_AZURE_150:min_node_count is 0 above -- checkov's static
  # analyzer can't resolve it through the for_each/local map indirection.
  # checkov:skip=CKV_AZURE_142:local auth is needed for az ml job submission from CI
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

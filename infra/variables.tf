variable "location" {
  description = "Azure region. westus3 is the only scanned region with A100 SKUs open to this subscription."
  type        = string
  default     = "westus3"
}

variable "project" {
  type    = string
  default = "minigpt"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "enable_a100" {
  description = "Create the 1x A100 smoke/sweep cluster (gpu-a100). Requires NCADSA100v4 quota granted in var.location — Azure ML validates quota at creation even for min_node_count = 0."
  type        = bool
  default     = false
}

variable "enable_a100_x4" {
  description = "Create the 4x A100 main training cluster (gpu-a100-x4). Requires NCADSA100v4 quota granted in var.location."
  type        = bool
  default     = false
}

variable "enable_t4" {
  description = "Create the 4x T4 fallback cluster (gpu-t4-x4). Flip this on if A100 quota is refused."
  type        = bool
  default     = false
}

variable "container_app_min_replicas" {
  type    = number
  default = 1
}

variable "monthly_budget_usd" {
  type    = number
  default = 190
}

variable "budget_alert_emails" {
  description = "Required non-empty — the consumption budget API rejects a notification with no contact channel at all."
  type        = list(string)
  default     = ["chiranjeevichary17@gmail.com"]
}

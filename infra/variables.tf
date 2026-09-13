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
  description = "Create the A100 clusters. Requires NCADSA100v4 quota granted in var.location."
  type        = bool
  default     = false
}

variable "enable_t4" {
  description = "Create the T4 fallback cluster. Flip this on if A100 quota is refused."
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
  type    = list(string)
  default = []
}

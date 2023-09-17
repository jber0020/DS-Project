provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "elec_rg" {
  name     = "elecrg"
  location = "Australia Southeast"
}

resource "azurerm_container_registry" "elec_acr" {
  name                     = "elecacr"
  resource_group_name      = azurerm_resource_group.elec_rg.name
  location                 = azurerm_resource_group.elec_rg.location
  sku                      = "Basic"
  admin_enabled            = true
  public_network_access_enabled = true
}


resource "azurerm_resource_group" "elec_rg" {
  name     = "elecrg"
  location = "Australia Southeast"
}

resource "azurerm_container_registry" "elec_acr" {
  name                     = "elecacr"
  resource_group_name      = azurerm_resource_group.elec_rg.name
  location                 = azurerm_resource_group.elec_rg.location
  sku                      = "Basic"
  admin_enabled            = true
  public_network_access_enabled = true
}

resource "azurerm_app_service_plan" "elec_asp" {
  name                = "elec-asp"
  location            = azurerm_resource_group.elec_rg.location
  resource_group_name = azurerm_resource_group.elec_rg.name

  sku {
    tier = "Standard"
    size = "S1"
  }
}

resource "azurerm_app_service" "elec_app_service" {
  name                = "elec-app-service"
  location            = azurerm_resource_group.elec_rg.location
  resource_group_name = azurerm_resource_group.elec_rg.name
  app_service_plan_id = azurerm_app_service_plan.elec_asp.id

  site_config {
    app_command_line = ""
    linux_fx_version = "DOCKER|${azurerm_container_registry.elec_acr.login_server}/flaskapp:latest"
  }

  app_settings = {
    "DOCKER_REGISTRY_SERVER_URL"      = "https://${azurerm_container_registry.elec_acr.login_server}"
    "DOCKER_REGISTRY_SERVER_USERNAME" = azurerm_container_registry.elec_acr.admin_username
    "DOCKER_REGISTRY_SERVER_PASSWORD" = azurerm_container_registry.elec_acr.admin_password
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_mysql_server" "elec_mysql_server" {
  name                = "elec-mysql-server"
  location            = azurerm_resource_group.elec_rg.location
  resource_group_name = azurerm_resource_group.elec_rg.name

  administrator_login          = "mysqladmin"
  administrator_login_password = "H@sh1Pa$$w0rd" # Change this to a secure password.

  sku_name   = "B_Gen5_2"
  storage_mb = 5120
  version    = "5.7"

  public_network_access_enabled = true
  ssl_enforcement_enabled       = false
}

resource "azurerm_mysql_database" "elec_mysql_database" {
  name                = "elecdatabase"
  resource_group_name = azurerm_resource_group.elec_rg.name
  server_name         = azurerm_mysql_server.elec_mysql_server.name
  charset             = "utf8"
  collation           = "utf8_unicode_ci"
}

resource "azurerm_mysql_firewall_rule" "elec_mysql_fw" {
  name                = "AllowAll"
  resource_group_name = azurerm_resource_group.elec_rg.name
  server_name         = azurerm_mysql_server.elec_mysql_server.name
  start_ip_address    = "0.0.0.0"
  end_ip_address      = "255.255.255.255"
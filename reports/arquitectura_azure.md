# Arquitectura en la nube (Azure)

Arquitectura propuesta para estandarizar la solución:
ingesta, almacenamiento, procesamiento, entrenamiento/registro de modelo, exposición de resultados, hosting del agente, orquestación, secretos y monitoreo.

## Diagrama Arquitectura Propuesta:

![Arquitectura objetivo](figures/arquitectura.png)

## Componentes y responsabilidades

| Capa | Servicio Azure | Rol en esta solución |
|---|---|---|
| Orquestación / ingesta | **Azure Data Factory** | Programa la descarga de X, Y, Z y precios de equipos; dispara el pipeline (`data→eda→features→train→evaluate→forecast`) y el reentrenamiento periódico. |
| Almacenamiento | **ADLS Gen2** (Bronze/Silver/Gold) | Bronze = CSV crudos (formatos sucios); Silver = series parseadas/reconciliadas y agregadas a mensual; Gold = `reports/artifacts/*` (selección, métricas, pronósticos). |
| Procesamiento + ML | **Azure Databricks** o **Azure ML** | Ejecuta el paquete `cost_forecast` (mismo código). EDA, votación por equipo, VIF, comparación de modelos, proyección Monte Carlo. |
| Entrenamiento / registro | **Azure ML Workspace + Model Registry** | Versiona modelos (ElasticNet/SARIMAX), métricas y datasets; habilita rollback y linaje. |
| Exposición | **Azure Container Apps** / **App Service** | API REST que sirve pronósticos y bandas a planeación financiera. |
| Agente | **Container Apps** + **API de Anthropic** | Hospeda el agente LangGraph; lee artefactos (Gold) y consulta la web para contexto de mercado. |
| Secretos | **Azure Key Vault** | `ANTHROPIC_API_KEY`, cadenas de conexión; inyectados vía *Managed Identity* (sin secretos en código). |
| Identidad | **Microsoft Entra ID** | RBAC y *Managed Identities* para acceso sin contraseñas entre servicios. |
| Monitoreo | **Application Insights + Azure Monitor** | Latencia/errores de API y agente, *data/Model drift*, alertas de degradación de MAPE. |

## Flujo operativo (MLOps)

1. **Ingesta programada** (ADF) deja los crudos en *Bronze*.
2. **Job de procesamiento** (Databricks/Azure ML) corre el pipeline; escribe *Silver* y *Gold* y registra el modelo en Azure ML.
3. **API** sirve los pronósticos desde *Gold*/Model Registry.
4. **Agente** (Container App) responde preguntas combinando *Gold* + búsqueda web.
5. **Monitoreo** vigila drift y error; si el MAPE supera un umbral, ADF dispara reentrenamiento (CI/CD con Azure DevOps/GitHub Actions).

> Equivalencias multi-nube: ADLS↔S3/GCS, Azure ML↔SageMaker/Vertex AI,
> Container Apps↔ECS-Fargate/Cloud Run, Key Vault↔Secrets Manager/Secret Manager,
> Data Factory↔Glue/Cloud Composer, App Insights↔CloudWatch/Cloud Monitoring.

---

## Demo implementado en Azure

### ¿Qué se desplegó?

Se desplegaron **3 servicios** en la región `eastus`, grupo de recursos `rg-analisis-costos`:

| # | Servicio | Nombre / URL |
|---|---|---|
| 1 | **Azure Container Registry** | `acranalisisocostos.azurecr.io` |
| 2 | **Container App — API** | `cost-api.redsky-57888008.eastus.azurecontainerapps.io` |
| 3 | **Container App — UI** | `cost-ui.redsky-57888008.eastus.azurecontainerapps.io` |

- Imágenes construidas con `--platform linux/amd64`.
- `ANTHROPIC_API_KEY` inyectada como *secret* del Container App (no Key Vault en el demo).
- Artefactos (`reports/artifacts/`, `reports/figures/`), sin almacenamiento externo.
- Escala mínima: 0 réplicas.

### Diagrama demo implementado

![Demo implementado](figures/demo.png)

### Diferencias con la arquitectura objetivo

| Aspecto | Demo implementado | Arquitectura objetivo |
|---|---|---|
| Secretos | `--secrets` en Container App (env var) | Azure Key Vault + Managed Identity |
| Artefactos / modelos | Horneados en la imagen Docker | ADLS Gen2 (Gold) + Azure ML Model Registry |
| Ingesta de datos | Manual (incluidos en imagen) | Azure Data Factory (pipelines programados) |
| Procesamiento ML | Local (`make all`) | Azure Databricks / Azure ML |
| Observabilidad | Logs nativos de Container Apps | Application Insights + Azure Monitor |
| Reentrenamiento | Manual | ADF dispara job ante drift de MAPE |

### Pasos de despliegue reales (ejecutados)

```bash
# Variables
RESOURCE_GROUP=rg-analisis-costos
LOCATION=eastus
ACR_NAME=acranalisisocostos
ENV_NAME=env-analisis-costos

# Registro de proveedor (solo primera vez)
az provider register -n Microsoft.OperationalInsights --wait

# ACR
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
ACR_USER=$(az acr credential show -n $ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

# Build multiplataforma (requerido desde Mac ARM)
docker build --platform linux/amd64 -f Dockerfile.api -t $ACR_NAME.azurecr.io/cost-api:v2 .
docker build --platform linux/amd64 -f Dockerfile.ui  -t $ACR_NAME.azurecr.io/cost-ui:latest .
docker push $ACR_NAME.azurecr.io/cost-api:v2
docker push $ACR_NAME.azurecr.io/cost-ui:latest

# Entorno
az containerapp env create --name $ENV_NAME --resource-group $RESOURCE_GROUP --location $LOCATION

# API
az containerapp create --name cost-api --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME --image $ACR_NAME.azurecr.io/cost-api:v2 \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USER --registry-password $ACR_PASS \
  --target-port 8000 --ingress external \
  --secrets anthropic-key=<ANTHROPIC_API_KEY> \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key \
  --min-replicas 0 --max-replicas 2 --cpu 0.5 --memory 1.0Gi

# UI
API_URL=cost-api.redsky-57888008.eastus.azurecontainerapps.io
az containerapp create --name cost-ui --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME --image $ACR_NAME.azurecr.io/cost-ui:latest \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USER --registry-password $ACR_PASS \
  --target-port 8501 --ingress external \
  --env-vars API_BASE_URL=https://$API_URL \
  --secrets anthropic-key=<ANTHROPIC_API_KEY> \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key \
  --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi
```

> Equivalencias: ACR↔ECR/Artifact Registry, Container Apps↔ECS-Fargate/Cloud Run,
> Key Vault↔Secrets Manager/Secret Manager.
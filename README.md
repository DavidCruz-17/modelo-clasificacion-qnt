# Proyecto QNT – Clasificación de clientes

Este proyecto busca identificar en qué segmentos QNT debe aplicar mayor intensidad para mejorar la efectividad de gestión de cartera castigada.

## Ejecución local

1. Crear entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate en Windows
   ```

2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Ejecutar el modelo:
   ```
   python modelo_qnt_despliegue.py
   ```

## DVC

Para cargar los datos necesarios:
```
dvc pull
```


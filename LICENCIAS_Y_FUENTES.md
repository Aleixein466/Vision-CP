# Licencias, fuentes y procedencia

**Consulta:** 2026-10-25. Este registro técnico no sustituye asesoría jurídica.

## Datos descargados

| Recurso | Procedencia | Integridad | Estado legal/técnico |
|---|---|---|---|
| COCO128 ZIP | `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip` | SHA-256 `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e` | Inspeccionado; solo piloto interno. El `LICENSE` incluido corresponde al software GPL-3.0 y no demuestra derechos de cada foto. |
| YOLO26n | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt` | SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Peso oficial descargado por Ultralytics 8.4.189. Revisar [licencia Ultralytics](https://www.ultralytics.com/license) antes de distribución/comercialización. |

El manifiesto de descarga de COCO128 está en `datasets/manifests/coco128.zip.download.json`; cada imagen procesada tiene hash y origen en `datasets/manifests/visionar_pilot_v1.csv`.

## Reglas por fuente

- COCO: [términos oficiales](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). COCO no posee las imágenes; antes de redistribuir hay que recuperar y revisar licencia/URL por imagen desde los metadatos oficiales.
- Open Images V7: [anotaciones CC BY 4.0; imágenes listadas CC BY 2.0](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), pero Google no garantiza el estado de cada imagen. Exigir revisión por archivo.
- Poly Haven: [todos los activos son CC0](https://polyhaven.com/license). Atribución no obligatoria pero se conservará por trazabilidad.
- Smithsonian Open Access: activos marcados CC0 se pueden reutilizar; [formatos 3D incluyen glTF/GLB/OBJ](https://www.si.edu/openaccess/faq). No asumir CC0 si una ficha dice “Usage Conditions Apply”.
- Sketchfab: licencia por activo. La [API de descarga](https://sketchfab.com/developers/download-api) requiere autenticación y su guía exige mostrar atribución para CC BY.
- Khronos Sample Assets: revisar el README/licencia de cada activo; la licencia no es uniforme en todo el repositorio.

## Política de aprobación

Un recurso pasa de candidato a aprobado solo si tiene URL canónica, autor, licencia explícita, versión o fecha, hash del archivo, restricciones documentadas y revisión de contenido. Licencia ausente o ambigua implica `blocked_license`; no se descarga. Ningún activo 3D fue descargado en esta fase.


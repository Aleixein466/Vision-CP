# VisionAR AI - seguimiento de escena y entrenamiento

**Fecha:** 25 de octubre de 2026.

## Cambios implementados

- Seguimiento temporal por categoria para camara y video.
- Una deteccion debe aparecer en dos inferencias consecutivas para entrar en escena.
- Mientras permanece visible, conserva su caja, nombre, confianza y ubicacion sin repetir el aviso.
- Tolera perdidas breves: solo sale despues de tres inferencias ausentes y al menos 1,5 segundos.
- Tras una salida confirmada, el mismo tipo de objeto puede anunciarse de nuevo al regresar.
- Nuevo modo **Billetera y contenidos**, con busqueda de billetera, billetes, tarjetas, llaves y memoria USB.
- El VLM recibe instrucciones para describir unicamente contenido visible, sin inventar valores, identidad ni datos de tarjetas.
- Dataset propio ampliado a ocho clases: telefono, perfume, llaves, USB, pez, billetera, billete y tarjeta.

## Validacion ejecutada

- `npm test`: 10 de 10 pruebas superadas.
- `npm run tracking:test`: secuencia entrada/permanencia/salida/reentrada superada.
- API YOLO26-seg real: 2 detecciones y 2 mascaras sobre una imagen de prueba real.
- Catalogo: 10 combinaciones (nano, small, medium, large y extra large; deteccion y segmentacion).
- Inferencia HTTP YOLO26n-seg caliente: 77,78 ms, equivalente teorico a 12,86 FPS, sin contar captura y dibujo.
- Modo contenidos YOLOE caliente: 115,92 ms, equivalente teorico a 8,63 FPS sobre la imagen real de control. No detecto contenidos porque esa imagen contiene un portatil y un raton, resultado esperado para el filtro.
- El error 502 registrado provenia de agotamiento de RAM de PyTorch. Al cambiar entre deteccion general y especializada ahora se libera la familia de modelos anterior, y la cola del navegador sigue sin solapar solicitudes.
- El servidor de prueba se ejecuto en el puerto 8180 y se cerro al finalizar. El servidor del usuario en 8080/8091 no fue interrumpido.

## Resultado del intento de entrenamiento

Se ejecuto `python scripts/train_visionar.py --execute`. El auditor detuvo el proceso correctamente porque el dataset propio tiene 0 imagenes y 0 anotaciones reales en sus ocho clases. No se generaron pesos ni metricas ficticias.

Requisito minimo configurado: 30 anotaciones revisadas por clase. Objetivo recomendado: 100 a 200 imagenes variadas por clase, manteniendo cada sesion fisica en un solo conjunto para evitar fuga entre train, val y test.

Cuando los datos esten listos, `npm run train:own` entrenara `yolo26n-seg.pt` en CPU con 40 epocas, imagen de 512 px, lote 2, `workers=0` y parada temprana. Antes de iniciar volvera a auditar cobertura, duplicados y revision de mascaras.

## Limitaciones pendientes

- Sin fotografias reales de la billetera del usuario no es posible validar todavia billetera/billete/tarjeta ni calcular precision, recall o mAP de esas clases.
- El seguimiento actual mantiene identidad por categoria. No vuelve a anunciar una billetera estable; si se necesitan varias billeteras de la misma clase con identidades separadas, la siguiente mejora seria seguimiento por instancia con IoU/ByteTrack.
- Las escalas large y extra large pueden ser demasiado lentas para tiempo real en CPU; nano-seg es la opcion recomendada para la camara actual.

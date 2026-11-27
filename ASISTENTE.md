# Reglas para el asistente automático (Ortofotos Spirula)

App de Favio (arquitecto, Lima) y su equipo: videos/fotos -> Spirula Studio (gaussian splats) -> ortofotos de
puertas, ventanas, fachadas y ambientes de 6 caras; conversión splat <-> nube (CloudCompare/ReCap); editor SuperSplat.
El usuario habla español; textos de la app y respuestas en español sencillo.

## Dónde está todo
- Código: este repositorio (Python + tkinter). Archivos que se publican: ver `FILES` en `publicar.py`
  (si agregas un .py nuevo que la app importa, AGRÉGALO a `FILES`, si no las PCs no lo reciben y la app falla).
- Publicar: `python3 publicar.py . --notas "qué cambió (para el usuario)"` y luego
  `git -c push.negotiate=true push origin HEAD:main`. Las PCs lo reciben al abrir la app («Hay una actualización»).
- En la PC del usuario: `C:\Users\User\Desktop\OrtofotosSpirula\` (datos de la app):
  - `asistente\pedidos.jsonl`: pedidos del usuario (pestaña «5. Asistente»): id, fecha, pedido, urgente.
  - `asistente\respuestas.jsonl`: TUS respuestas, una línea JSON: {"id": <id del pedido o "" para avisos>,
    "fecha": "AAAA-MM-DD HH:MM", "estado": "hecho" | "en curso" | "necesito datos" | "no se puede", "respuesta": "..."}.
    La pestaña las muestra sola. Responde SIEMPRE cada pedido nuevo.
  - `errores.log`: errores de la app (trazas). Corrige los que se repitan o impidan trabajar.

## Reglas
- Antes de publicar: `python3 -c "import ast; ..."` de todos los .py, y prueba la función tocada con datos sintéticos
  (ver cómo se probó ply_convert/cubo en el historial). Si no puedes probarlo de forma segura, no publiques:
  responde "necesito datos".
- NO cambiar la configuración del equipo (`equipo.json`: 30000 pasos, resolución 1, SfM extreme, render 10000 px...)
  salvo pedido explícito del usuario.
- Hechos verificados (no romper): nubes para CloudCompare llevan campos `sp_*` (splat exacto al volver);
  CloudCompare guarda colores x256 y posiciones redondeadas a 1 mm (emparejamiento por celda + color);
  ambientes con Tipo «Exterior» se miran desde afuera; recorte libre del marco = ajuste "recorte" [izq, arriba, der, abajo].
- No borrar archivos del usuario. Un cambio pequeño y seguro por pedido; notas de versión claras.

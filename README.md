# Blender LOD BrickModel Generator

Blender-Add-on (BrickWars LEGO Optimizer), das importierte LEGO-Modelle (z. B. aus LDD oder BrickLink Studio) automatisch als Low-Poly-/LOD-Version nachbaut. Die Details des Originals werden als Texturen auf das neue Modell gebacken. Gedacht ist es vor allem für Game-Modding, etwa für Star Wars Battlefront II (2005).

## Funktionen

- Baut ausgewählte Modelle mit wenigen einfachen Formen neu auf, aus Einzelteilen oder aus einem zusammengefügten Mesh.
- Die gewünschte maximale Anzahl an Dreiecken ist einstellbar und wird eingehalten.
- Diffuse- und Normal-Map werden vom High-Poly-Modell in einen gemeinsamen Textur-Atlas gebacken.
- Noppen lassen sich als Normal Map übernehmen, entfernen oder als Sprites darstellen.
- Gleiche Teile teilen sich einen Texturbereich, dadurch wird die Textur schärfer.
- Verdeckte Flächen und Teile werden entfernt.
- Es gibt zwei Looks: Studio-Render oder heller Spielzeug-Look wie in LEGO-Spielen.
- Eigene Low-Poly-Meshes können ebenfalls mit Texturen vom Original bebacken werden.

Das Originalmodell bleibt unverändert. Das Ergebnis landet als `NAME_LOWPOLY` in einer eigenen Collection.

## Installation

Benötigt **Blender 5.2** oder neuer.

1. `BrickWars_LEGO_Optimizer-2.7.0.zip` herunterladen.
2. In Blender **Edit → Preferences → Get Extensions → ⋯ → Install from Disk** öffnen und die ZIP-Datei auswählen.
3. Im 3D-Viewport mit **N** die Seitenleiste öffnen und den Tab **BrickWars Optimizer** wählen.

## Benutzung

1. Modell im Objektmodus auswählen.
2. Dreiecksbudget und Optionen einstellen.
3. **Optimize Selection** klicken.

## Lizenz

GPL-3.0-or-later, siehe [LICENSE](LICENSE).

"""3D View controls and staged operator lifecycle."""
import traceback

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

from . import mesh, pipeline


class BrickWarsSettings(PropertyGroup):
    method: EnumProperty(name="Method", items=[
        ('PARTS', "Rebuild Parts",
         "Replace every selected part with a box, cylinder, rounded bar or slope; "
         "holes and studs become texture"),
        ('REBUILD', "Blocks from Single Mesh",
         "Rebuild the model from oriented blocks and bake the detail into textures"),
        ('GUIDED', "Bake onto Rebuilt Meshes",
         "Use the active mesh as source and the other selected meshes as the rebuilt shape")])
    target: IntProperty(name="Target Triangles", default=2000, min=24, max=20000,
                        soft_min=100, soft_max=5000,
                        description="Upper limit for the rebuilt mesh. BF2 guide: vehicles 1500-2000, props up to 500, buildings 200-3000")
    detail: EnumProperty(name="Analysis Detail", default='128', items=[
        ('96', "Fast", "96 voxels along the longest side"),
        ('128', "Standard", "128 voxels along the longest side"),
        ('192', "Fine", "192 voxels along the longest side; slower, keeps thinner parts")])
    cull_hidden: BoolProperty(name="Remove Hidden Faces", default=True,
                              description="Drop faces and whole parts that cannot be seen from outside")
    share_textures: BoolProperty(name="Share Identical Parts", default=True,
                                 description="Identical parts (same shape and color) use one "
                                             "texture region, so every region gets more pixels")
    studs: EnumProperty(name="Studs", default='TEXTURE', items=[
        ('TEXTURE', "Normal Map",
         "Bake the studs into the normal map and the lighting of the atlas"),
        ('REMOVE', "Remove",
         "Take the studs off before baking; the tops stay flat"),
        ('SPRITES', "Layered Sprites",
         "Take the studs off and draw the visible ones as two stacked, alpha-tested "
         "panels with a repeating stud texture per color, like older console LEGO "
         "games. A whole stud field costs 4 triangles, taken from the target")])
    chamfers: BoolProperty(name="Bevels and Slopes", default=True,
                           description="Allow 45 degree cuts on blocks")
    generate_uv: BoolProperty(name="Generate UV Map", default=True)
    resolution: EnumProperty(name="Texture Resolution", default='2048',
                             items=[(str(n), f"{n} × {n}", "") for n in (512, 1024, 2048, 4096)])
    uv_margin: FloatProperty(name="UV Island Margin", default=.007, min=.001, max=.05,
                             precision=3, subtype='FACTOR')
    pack: BoolProperty(name="Pack UV Islands", default=True)
    bake: BoolProperty(name="Bake Textures", default=True)
    lighting: FloatProperty(name="Baked Lighting", default=.85, min=0, max=1, subtype='FACTOR',
                            description="Bake ambient occlusion and soft studio light into "
                                        "the diffuse atlas for a rendered LEGO look")
    look: EnumProperty(name="Look", default='STUDIO', items=[
        ('STUDIO', "Studio Render",
         "Soft studio light and deep ambient occlusion, like a photoreal brick render"),
        ('GAME', "Bright Game",
         "Clean, saturated toy look of console LEGO games: bright light, light occlusion "
         "with crisp crevice lines, strong narrow highlights and more gloss")])
    edge_rounding: FloatProperty(name="Rounded Edges", default=1.0, min=0, max=4,
                                 description="Bake softly rounded plastic edges into the normal "
                                             "map. Zero keeps the edges sharp")
    edge_highlight: FloatProperty(name="Edge Highlights", default=1.0, min=0, max=2,
                                  subtype='FACTOR',
                                  description="Brighten rounded edges and stud rims in the "
                                              "diffuse atlas, like a studio render")
    specular_alpha: BoolProperty(name="Specular Mask in Alpha", default=True,
                                 description="Write the plastic gloss into the diffuse alpha. "
                                             "SWBF2 uses it when the material has the Specular "
                                             "flag; turn off if the alpha means transparency")
    ray_distance: FloatProperty(name="Projection Distance", default=0, min=0, precision=5,
                                description="Blender units. Zero derives the distance from the blocks")
    directory: StringProperty(name="Texture Directory", default="//BrickWars_Textures", subtype='DIR_PATH')
    progress: FloatProperty(default=0, min=0, max=1, options={'SKIP_SAVE'})
    stage: StringProperty(options={'SKIP_SAVE'})


class BRICKWARS_OT_optimize(Operator):
    bl_idname = "brickwars.optimize"
    bl_label = "Optimize Selection"
    bl_options = {'REGISTER'}
    running = False

    @classmethod
    def poll(cls, context):
        return (not cls.running and context.mode == 'OBJECT' and bool(pipeline.selected(context)))

    def begin(self, context):
        self.settings = context.scene.brickwars_settings
        self.generator = pipeline.run(context, self.settings)
        self.timer = None
        type(self).running = True
        context.window_manager.progress_begin(0, 100)

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)
        self.begin(context)
        self.timer = context.window_manager.event_timer_add(.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        self.begin(context)
        while True:
            status = self.advance(context)
            if status:
                return status

    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            self.report({'INFO'}, "Optimization cancelled.")
            return {'CANCELLED'}
        if event.type == 'TIMER':
            return self.advance(context) or {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def advance(self, context):
        try:
            self.settings.progress, self.settings.stage = next(self.generator)
            context.window_manager.progress_update(self.settings.progress*100)
            if context.area:
                context.area.tag_redraw()
        except StopIteration as finished:
            result = finished.value
            self.cleanup(context)
            before = result["brickwars_triangles_before"]
            after = result["brickwars_triangles_after"]
            level = 'INFO' if result["brickwars_target_met"] else 'WARNING'
            message = f"{before:,} → {after:,} triangles ({1-after/before:.1%} reduction)"
            if level == 'WARNING':
                message += ". Target exceeded with current settings."
            self.report({level}, message)
            return {'FINISHED'}
        except Exception as error:
            traceback.print_exc()
            self.cleanup(context)
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

    def cleanup(self, context):
        try:
            self.generator.close()
        finally:
            if self.timer:
                context.window_manager.event_timer_remove(self.timer)
                self.timer = None
            context.window_manager.progress_end()
            self.settings.stage = ""
            type(self).running = False
            if context.area:
                context.area.tag_redraw()

    def cancel(self, context):
        self.cleanup(context)


class BRICKWARS_PT_panel(Panel):
    bl_idname = "BRICKWARS_PT_panel"
    bl_label = "BrickWars Optimizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "BrickWars Optimizer"

    def draw(self, context):
        settings = context.scene.brickwars_settings
        layout = self.layout
        if BRICKWARS_OT_optimize.running:
            layout.progress(factor=settings.progress, type='BAR', text=settings.stage)
            layout.label(text="Esc cancels between stages")
            return
        stats = mesh.analyze(pipeline.selected(context))
        box = layout.box()
        box.label(text=f"{stats['triangles']:,} triangles · {stats['objects']} meshes")
        box.prop(settings, "method")
        if settings.method == 'GUIDED':
            box.label(text="Active mesh is the high-poly source")
        elif settings.method == 'PARTS':
            box.label(text="One selected object per LEGO part")
            box.prop(settings, "target")
            box.prop(settings, "cull_hidden")
            box.prop(settings, "share_textures")
        else:
            box.prop(settings, "target")
            box.prop(settings, "detail")
            box.prop(settings, "chamfers")
        box.prop(settings, "studs")
        obj = context.active_object
        if obj and "brickwars_triangles_before" in obj:
            before = obj["brickwars_triangles_before"]
            after = obj["brickwars_triangles_after"]
            box.label(text=f"Last result: {before:,} → {after:,}")
            if "brickwars_blocks" in obj:
                box.label(text=f"{obj['brickwars_blocks']} blocks")
            if "brickwars_stud_sprites" in obj:
                box.label(text=f"{obj['brickwars_stud_sprites']} studs in "
                               f"{obj.get('brickwars_stud_panels', 0)} panels")
        box = layout.box()
        box.prop(settings, "generate_uv")
        if settings.generate_uv:
            box.prop(settings, "uv_margin")
            box.prop(settings, "pack")
        box.prop(settings, "bake")
        if settings.bake:
            box.prop(settings, "resolution")
            box.prop(settings, "look")
            box.prop(settings, "lighting")
            box.prop(settings, "edge_rounding")
            if settings.lighting > 0:
                box.prop(settings, "edge_highlight")
            box.prop(settings, "specular_alpha")
            box.prop(settings, "ray_distance")
            box.prop(settings, "directory")
        layout.operator("brickwars.optimize", icon='MOD_REMESH')


CLASSES = (BrickWarsSettings, BRICKWARS_OT_optimize, BRICKWARS_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.brickwars_settings = PointerProperty(type=BrickWarsSettings)


def unregister():
    del bpy.types.Scene.brickwars_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

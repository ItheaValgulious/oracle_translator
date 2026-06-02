# 言灵炼金师 v1 -- 可配置常量

# ── Viewport ── (matches engine demo: 640x380 cells, cell_scale=4)
import os

VIEWPORT_WIDTH = 640
VIEWPORT_HEIGHT = 380
CELL_SCALE = 4              # px per cell (demo uses 4)
CELL_SCALE_OPTIONS = (2, 3, 4, 6, 8)

# ── World ── (proportional to viewport, same k-factors as old plan)
# Old: BIOME_WIDTH = 50 * 160 = 8000, WORLD_HEIGHT = 10 * 96 = 960
# New: BIOME_WIDTH = 50 * 640 = 32000, WORLD_HEIGHT = 10 * 380 = 3800
BIOME_WIDTH = 32000          # 50 * VIEWPORT_WIDTH
WORLD_HEIGHT = 3800          # 10 * VIEWPORT_HEIGHT
WORLD_WIDTH = BIOME_WIDTH * 4
CHUNK_SIZE = 320
CHUNK_CACHE_PREFETCH_X = 3
CHUNK_CACHE_PREFETCH_Y = 2
GPU_CHUNK_SAVE_DIR = "artifacts/gpu_chunks"
SAVE_NAME_ENV_VAR = "ORACLE_TRANSLATOR_SAVE_NAME"
DEFAULT_SAVE_NAME = "default"

# ── Hero ── (height=17, width keeps same 5:10 ratio → width=8.5)
HERO_WIDTH = 8.5             # 17 * (5/10) = 8.5
HERO_HEIGHT = 17.0           # cells
HERO_MAX_HP = 100.0
HERO_MAX_MP = 50.0
MP_REGEN_PER_SEC = 3.0
HERO_GRAVITY = 40.0
HERO_WALK_SPEED = 36.0
HERO_JUMP_VELOCITY = 36.88
HERO_PLACEHOLDER_DAMAGE_SCALE = 30.0

# ── Enemy A (archer) ──
ENEMY_A_WIDTH = 6.0
ENEMY_A_HEIGHT = 12.0
ENEMY_A_DETECT_RANGE = 80.0       # cells
ENEMY_A_ARROW_SPEED = 8.0        # cells/sec
ENEMY_A_ARROW_DAMAGE = 10.0      # HP
ENEMY_A_ARROW_INTERVAL = 1.5     # sec
ENEMY_A_PATROL_SPEED = 2.0      # cells/sec
ENEMY_A_PATROL_RANGE = 60.0     # cells
ENEMY_A_PATROL_MIN_DURATION = 1.5
ENEMY_A_PATROL_MAX_DURATION = 4.0

# ── Enemy B (flying bomber) ──
ENEMY_B_WIDTH = 8.0
ENEMY_B_HEIGHT = 8.0
ENEMY_B_DETECT_RANGE = 50.0
ENEMY_B_HOVER_HEIGHT = 25.0      # cells above ground
ENEMY_B_DRIFT_SPEED = 0.3        # cells/sec
ENEMY_B_CHARGE_SPEED = 8.0      # cells/sec
ENEMY_B_EXPLOSION_RADIUS = 6.0  # cells
ENEMY_B_EXPLOSION_DAMAGE = 25.0 # HP

# ── Spell damage ──
SPELL_BASE_DAMAGE = 15.0        # HP per tick to enemies in brush radius

# ── Enemy C (boss) ──
ENEMY_C_WIDTH = 16.0
ENEMY_C_HEIGHT = 24.0
ENEMY_C_MAX_HP = 200.0
ENEMY_C_ATTACK_CYCLE = 3.0       # sec
BOSS_FIREBALL_SPEED = 10.0      # cells/sec
BOSS_FIREBALL_RADIUS = 4.0      # cells
BOSS_OIL_SPRAY_WIDTH = 10.0     # cells
BOSS_COLLAPSE_WIDTH = 10.0      # cells
BOSS_COLLAPSE_HEIGHT = 20.0     # cells

# ── Spawn distribution (seed-based continuous) ──
ENEMY_SPACING = 200           # check every 200 world columns for enemy spawn
ENEMY_TYPE_SALT_A = 800       # hash salt for type A spawn check
ENEMY_TYPE_SALT_B = 810       # hash salt for type B spawn check
ENEMY_TYPE_SALT_C = 820       # hash salt for type C (boss)
ENEMY_A_PROBABILITY = 0.25    # probability at each spawn point
ENEMY_B_PROBABILITY = 0.10    # probability at each spawn point

# ── Enemy HP ──
ENEMY_A_MAX_HP = 30.0
ENEMY_B_MAX_HP = 15.0

# ── Projectile physics ──
ARROW_SPEED = 25.0            # cells/sec (fast parabolic arc)
ARROW_GRAVITY = 40.0          # cells/sec² (matches hero gravity for consistent feel)
ARROW_FLIGHT_TIME = 0.8       # sec (time to reach target, used for trajectory solve)
ARROW_DAMAGE = 10.0           # HP per arrow hit
BOSS_FIREBALL_SPEED = 20.0    # cells/sec
BOSS_FIREBALL_DAMAGE = 30.0   # HP per fireball hit
BOSS_FIREBALL_MAX_AGE = 4.0   # sec before fireball expires

# ── Enemy movement ──
ENEMY_SLOPE_MAX_CLIMB = 4     # max cells an enemy can climb in one step
ENEMY_DESTUCK_LIFT = 1.0

# ── Explosion ──
BLAST_PRESSURE = 140.0        # injected pressure (normal air = 1.0)
BLAST_PRESSURE_RADIUS = 7     # cells
BLAST_FIRE_RADIUS = 8         # cells (visual fireball area)
BLAST_PRESSURE_BURST_TICKS = 6
BLAST_PRESSURE_DECAY = 0.8
BLAST_PRESSURE_CORE_SCALE = 0.42
BLAST_PRESSURE_SHELL_SCALE = 0.95
BLAST_PRESSURE_SHELL_WIDTH = 4
BLAST_PRESSURE_SHELL_SPEED = 3.0
BLAST_SECONDARY_PRESSURE_SCALE = 0.55
BLAST_GAS_RADIUS = 4
BLAST_RING_DISTANCE = 6
BLAST_GAS_VELOCITY = 24.0
BLAST_DEBRIS_VELOCITY = 18.0
EXPLOSION_DAMAGE = 25.0       # direct HP damage on EnemyB explosion

# ── Visual ──
DAMAGE_FLASH_DURATION = 0.15  # sec

# ── Plains biome ──
PLAINS_GROUND_BASE_Y = 3200       # ground level (old: 800 * (380/96) ≈ 3167 → round to 3200)
PLAINS_NOISE_SCALE = 200.0
PLAINS_NOISE_AMPLITUDE = 60.0
PLAINS_NOISE_OCTAVES = 4
PLAINS_NOISE_PERSISTENCE = 0.5
PLAINS_FLOOR_DEPTH = 160
PLAINS_GRASS_SURFACE_DEPTH = 5
PLAINS_TREE_HEIGHT = 40
PLAINS_TREE_CANOPY_W = 60
PLAINS_TREE_CANOPY_H = 40
PLAINS_POND_WIDTH = 80
PLAINS_POND_DEPTH = 30
PLAINS_POND_WATER_DEPTH = 10
PLAINS_SURFACE_PATCH_SCALE = 95.0
PLAINS_SURFACE_GRASS_THRESHOLD = 0.56
PLAINS_SURFACE_SAND_THRESHOLD = 0.28

# ── Hillside biome ──
HILLSIDE_GROUND_START_Y = 3200    # matches plains
HILLSIDE_GROUND_END_Y = 1280      # climbs ~1920 cells (old: 320*(380/96) ≈ 1267 → 1280)
HILLSIDE_NOISE_SCALE = 150.0
HILLSIDE_NOISE_AMPLITUDE = 40.0
HILLSIDE_NOISE_OCTAVES = 3
HILLSIDE_NOISE_PERSISTENCE = 0.5

# ── Alpine biome ──
ALPINE_ISLAND_COUNT = 14
ALPINE_ISLAND_MIN_WIDTH = 200
ALPINE_ISLAND_MAX_WIDTH = 500
ALPINE_ISLAND_MIN_HEIGHT = 15
ALPINE_ISLAND_MAX_HEIGHT = 30
ALPINE_ISLAND_Y_MIN = 800         # old: 200*(380/96) ≈ 792 → 800
ALPINE_ISLAND_Y_MAX = 2400        # old: 600*(380/96) ≈ 2375 → 2400
ALPINE_BRIDGE_THICKNESS = 8
ALPINE_ICE_SURFACE_DEPTH = 5
ALPINE_AMBIENT_TEMP = -10.0
ALPINE_SHAFT_X_RATIO = 0.8
ALPINE_SHAFT_WIDTH = 4

# ── Underground biome ──
UNDERGROUND_Y_START = 1920        # old: 480*(380/96) ≈ 1900 → 1920
UNDERGROUND_SURFACE_FIRE_DEPTH = 60
UNDERGROUND_SURFACE_POISON_DEPTH = 240
UNDERGROUND_STONE_CAP_DEPTH = 180
UNDERGROUND_CHAMBER_COUNT = 7
UNDERGROUND_CHAMBER_MIN_W = 80
UNDERGROUND_CHAMBER_MIN_H = 60
UNDERGROUND_CHAMBER_MAX_W = 150
UNDERGROUND_CHAMBER_MAX_H = 100
UNDERGROUND_CORRIDOR_WIDTH = 20
UNDERGROUND_CORRIDOR_HEIGHT = 16
UNDERGROUND_BOSS_CHAMBER_W = 200
UNDERGROUND_BOSS_CHAMBER_H = 150

# ── Snow system ──
SNOW_INTERVAL = 0.5    # sec
SNOW_PROBABILITY = 0.3
SNOW_TOP_Y = 0         # world top row

# ── Wood/Grass growth ──
GROW_INTERVAL = 5.0     # sec between growth attempts
GROW_MIN_INTEGRITY = 0.5
WOOD_MAX_GENERATION = 6
GRASS_MAX_GENERATION = 4

# ── Rendering ──
CELL_TEXTURE_SIZE = 16  # px
ENTITY_ANIM_FPS = 8

# ── Placeholder ──
PLACEHOLDER_HARDNESS = 2.0
PLACEHOLDER_RENDER_COLOR = (0, 0, 0)  # invisible in grid render

# ── STT (Speech-to-Text) ──
STT_MODEL_DIR = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
STT_SAMPLE_RATE = 16000
STT_NUM_THREADS = 4


def chunk_storage_root() -> str:
    save_name = os.environ.get(SAVE_NAME_ENV_VAR, DEFAULT_SAVE_NAME).strip() or DEFAULT_SAVE_NAME
    return os.path.join("storage", save_name)

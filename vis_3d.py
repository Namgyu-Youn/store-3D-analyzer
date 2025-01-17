import pyrender
import trimesh

# 메시 로드
mesh = trimesh.load("store_3d.obj")

# 렌더러 설정
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh))

# 뷰어 실행
pyrender.Viewer(scene, use_raymond_lighting=True)

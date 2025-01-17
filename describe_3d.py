import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def analyze_store_3d(obj_path: str = "store_3d.obj") -> None:
    """매장 3D 모델을 분석하고 정보를 출력합니다."""
    try:
        # OBJ 파일 로드
        print("\n=== Loading and Analyzing 3D Store Model ===")
        mesh = o3d.io.read_triangle_mesh(obj_path)

        # 기본 정보 출력
        print("\n1. Basic Mesh Information:")
        print(f"- Number of vertices: {len(mesh.vertices)}")
        print(f"- Number of triangles: {len(mesh.triangles)}")

        # 바운딩 박스 정보
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        print("\n2. Store Dimensions:")
        print(f"- Width (X): {max_bound[0] - min_bound[0]:.2f} meters")
        print(f"- Height (Y): {max_bound[1] - min_bound[1]:.2f} meters")
        print(f"- Length (Z): {max_bound[2] - min_bound[2]:.2f} meters")

        # 중심점 계산
        center = mesh.get_center()
        print(f"\n3. Store Center Point: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

        # 파일 정보
        file_info = Path(obj_path).stat()
        print("\n4. File Information:")
        print(f"- File size: {file_info.st_size / 1024:.2f} KB")
        print(f"- Last modified: {file_info.st_mtime}")

        # 메시 검증
        print("\n5. Mesh Validation:")
        print(f"- Has vertices: {mesh.has_vertices()}")
        print(f"- Has triangles: {mesh.has_triangles()}")
        print(f"- Has vertex normals: {mesh.has_vertex_normals()}")
        print(f"- Has vertex colors: {mesh.has_vertex_colors()}")
        print(f"- Is watertight: {mesh.is_watertight()}")

        # 추가적인 통계
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            print("\n6. Color Statistics:")
            print(f"- Unique colors: {len(np.unique(colors, axis=0))}")
            print(f"- Average RGB: ({np.mean(colors[:,0]):.2f}, " f"{np.mean(colors[:,1]):.2f}, {np.mean(colors[:,2]):.2f})")

    except Exception as e:
        print(f"Analysis error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    analyze_store_3d()

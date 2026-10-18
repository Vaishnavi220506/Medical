// 3D tumor viewer: renders marching-cubes meshes from /api/studies/{id}/mesh with three.js.
let viewer = null;

export function disposeViewer() {
  if (!viewer) return;
  cancelAnimationFrame(viewer.frame);
  viewer.resize.disconnect();
  viewer.renderer.dispose();
  viewer.container.innerHTML = "";
  viewer = null;
}

export async function renderMeshes(container, data) {
  disposeViewer();
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");

  const width = container.clientWidth, height = container.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070d);
  const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 5000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const light = new THREE.DirectionalLight(0xffffff, 1.1);
  light.position.set(1, 1, 2);
  scene.add(light);
  const back = new THREE.DirectionalLight(0xffffff, 0.4);
  back.position.set(-1, -1, -2);
  scene.add(back);

  const group = new THREE.Group();
  const handles = [];
  for (const m of data.meshes) {
    const geometry = new THREE.BufferGeometry();
    // Vertices come as (slice, row, col) in mm: slice -> superior, row -> posterior,
    // col -> patient left. Map so the default camera looks at the patient's face:
    // screen x = patient left, y = superior, z = anterior (towards the viewer).
    const v = m.vertices;
    const pos = new Float32Array(v.length);
    for (let i = 0; i < v.length; i += 3) {
      pos[i] = v[i + 2];
      pos[i + 1] = v[i];
      pos[i + 2] = -v[i + 1];
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geometry.setIndex(m.faces);
    geometry.computeVertexNormals();
    const material = new THREE.MeshPhongMaterial({
      color: new THREE.Color(`rgb(${m.color.join(",")})`),
      transparent: m.opacity < 1,
      opacity: m.opacity,
      depthWrite: m.opacity >= 1,
      side: THREE.DoubleSide,
      shininess: 40,
    });
    const mesh = new THREE.Mesh(geometry, material);
    group.add(mesh);
    handles.push(mesh);
  }
  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  group.position.sub(center);
  scene.add(group);
  const size = box.getSize(new THREE.Vector3()).length();
  camera.position.set(0, size * 0.15, size * 1.25);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.2;
  renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; });

  const resize = new ResizeObserver(() => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  resize.observe(container);

  viewer = { renderer, container, resize, frame: 0 };
  const tick = () => {
    controls.update();
    renderer.render(scene, camera);
    viewer.frame = requestAnimationFrame(tick);
  };
  tick();
  return handles;
}

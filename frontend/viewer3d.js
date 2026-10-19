// 3D tumor viewer (three.js): glowing "hologram" brain shell, glossy tumor sub-regions,
// studio lighting, a live slice plane synced with the 2D viewer, cut-away clipping,
// preset views, auto-rotate and screenshots.
let viewer = null;

export function disposeViewer() {
  if (!viewer) return;
  cancelAnimationFrame(viewer.frame);
  viewer.resize.disconnect();
  viewer.renderer.dispose();
  viewer.container.innerHTML = "";
  viewer = null;
}

// Fresnel "x-ray" shader: transparent in the middle, glowing at grazing angles.
const FRESNEL_VERTEX = `
  varying vec3 vNormal;
  varying vec3 vView;
  #include <clipping_planes_pars_vertex>
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mvPosition.xyz);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
  }`;
const FRESNEL_FRAGMENT = `
  uniform vec3 glowColor;
  uniform float strength;
  varying vec3 vNormal;
  varying vec3 vView;
  #include <clipping_planes_pars_fragment>
  void main() {
    #include <clipping_planes_fragment>
    float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.2);
    gl_FragColor = vec4(glowColor, clamp(rim * strength + 0.02, 0.0, 1.0));
  }`;

// Per sub-region look: edema is translucent so the core inside stays visible.
const STYLE = {
  edema: { opacity: 0.38, roughness: 0.5, renderOrder: 3 },
  enhancing_tumor: { opacity: 0.88, roughness: 0.28, renderOrder: 2 },
  necrotic_core: { opacity: 1.0, roughness: 0.45, renderOrder: 1 },
};

// opts.sliceUrl(i): image of axial slice i (3D scans); opts.imageUrl: the uploaded image (2D -> 3D preview).
export async function renderMeshes(container, data, opts = {}) {
  disposeViewer();
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");

  const width = container.clientWidth, height = container.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.localClippingEnabled = true;
  renderer.setClearColor(0x000000, 0);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x050914, 400, 900);
  const camera = new THREE.PerspectiveCamera(35, width / height, 1, 5000);

  // Studio lighting: soft sky/ground fill, warm key, cool fill, rim light from behind.
  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x101426, 0.9));
  const key = new THREE.DirectionalLight(0xfff1e0, 2.4);
  key.position.set(1.2, 1.6, 2);
  const fill = new THREE.DirectionalLight(0x7dd3fc, 0.9);
  fill.position.set(-2, 0.3, 1);
  const rim = new THREE.DirectionalLight(0xa78bfa, 1.6);
  rim.position.set(0, 1, -2.5);
  scene.add(key, fill, rim);

  // Vertices arrive as (slice, row, col) in mm: map to x = patient left, y = superior, z = anterior.
  const toGeometry = (m) => {
    const v = m.vertices;
    const pos = new Float32Array(v.length);
    for (let i = 0; i < v.length; i += 3) {
      pos[i] = v[i + 2];
      pos[i + 1] = v[i];
      pos[i + 2] = -v[i + 1];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(m.faces);
    g.computeVertexNormals();
    return g;
  };

  const group = new THREE.Group();
  const cutPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e6);
  const handles = [];
  for (const m of data.meshes) {
    const geometry = toGeometry(m);
    let mesh;
    if (m.name === "anatomy") {
      const material = new THREE.ShaderMaterial({
        uniforms: { glowColor: { value: new THREE.Color(0x67e8f9) }, strength: { value: 1.0 } },
        vertexShader: FRESNEL_VERTEX,
        fragmentShader: FRESNEL_FRAGMENT,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        clipping: true,
        clippingPlanes: [cutPlane],
      });
      mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = 10;
    } else {
      const style = STYLE[m.name] || { opacity: m.opacity ?? 0.9, roughness: 0.35, renderOrder: m.opacity < 0.5 ? 4 : 2 };
      const color = new THREE.Color(`rgb(${m.color.join(",")})`);
      const material = new THREE.MeshPhysicalMaterial({
        color,
        roughness: style.roughness,
        metalness: 0.05,
        clearcoat: 0.8,
        clearcoatRoughness: 0.25,
        sheen: 0.4,
        sheenColor: color.clone().offsetHSL(0, 0, 0.2),
        emissive: color.clone().multiplyScalar(0.12),
        transparent: style.opacity < 1,
        opacity: style.opacity,
        depthWrite: style.opacity >= 1,
        side: THREE.DoubleSide,
      });
      mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = style.renderOrder;
    }
    mesh.userData.targetOpacity = mesh.material.opacity ?? 1;
    mesh.userData.name = m.name;
    group.add(mesh);
    handles.push(mesh);
  }

  // Scan images in 3D: glowing "light-box" planes (additive blending makes the black background vanish).
  const loader = new THREE.TextureLoader();
  // Colour is added; alpha is left alone, so the (transparent) canvas does not turn opaque black
  // where the image is dark.
  const imageMaterial = (opacity) => new THREE.MeshBasicMaterial({
    transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.SrcAlphaFactor, blendDst: THREE.OneFactor,
    blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
  });
  const loadInto = (material, url) => loader.load(url, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    material.map?.dispose();
    material.map = tex;
    material.needsUpdate = true;
  });
  let imagePlane = null;
  if (data.image_plane && opts.imageUrl) {
    // 2D -> 3D preview: the uploaded image stands in the middle of the estimated volume.
    const [w, h] = data.image_plane.size;
    imagePlane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), imageMaterial(0.5));
    imagePlane.position.set(...data.image_plane.center);
    imagePlane.renderOrder = 0; // drawn before the tumor, so the tumor keeps its colour
    imagePlane.userData = { name: "image", targetOpacity: 0.5 };
    loadInto(imagePlane.material, opts.imageUrl);
    group.add(imagePlane);
    handles.push(imagePlane);
  }

  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  group.position.sub(center);
  scene.add(group);
  const radius = size.length() / 2;

  // Floor grid that fades into the background.
  const grid = new THREE.PolarGridHelper(radius * 1.6, 12, 6, 64, 0x1e3a5f, 0x13233d);
  grid.position.y = -size.y / 2 - radius * 0.08;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);

  // Slice plane: where the 2D viewer currently is, showing that slice's image (3D scans).
  const planeSize = data.slice_plane?.size || [size.x, size.z];
  const sliceMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(planeSize[0], planeSize[1]),
    opts.sliceUrl ? imageMaterial(0.9) : new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }),
  );
  sliceMesh.rotation.x = Math.PI / 2; // image top (row 0, anterior) towards +z
  const sliceEdge = new THREE.LineSegments(
    new THREE.EdgesGeometry(sliceMesh.geometry),
    new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.8 }),
  );
  sliceMesh.add(sliceEdge);
  sliceMesh.visible = false;
  sliceMesh.renderOrder = 6;
  group.add(sliceMesh);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.9;
  controls.minDistance = radius * 0.6;
  controls.maxDistance = radius * 6;
  renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; viewer?.onAutoRotate?.(false); });

  const views = {
    front: new THREE.Vector3(0, radius * 0.15, radius * 2.6),
    side: new THREE.Vector3(radius * 2.6, radius * 0.15, 0),
    top: new THREE.Vector3(0, radius * 2.6, 0.01),
    angled: new THREE.Vector3(radius * 1.6, radius * 1.1, radius * 1.9),
  };
  let flight = null; // camera animation
  const flyTo = (target, ms = 900) => {
    flight = { from: camera.position.clone(), to: target.clone(), start: performance.now(), ms };
  };
  camera.position.copy(views.angled).multiplyScalar(1.8);
  flyTo(views.angled, 1400);

  // Fade the meshes in on open.
  const born = performance.now();
  handles.forEach((m) => { if ("opacity" in m.material) { m.material.transparent = true; m.material.opacity = 0; } });

  const resize = new ResizeObserver(() => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  resize.observe(container);

  const voxel = data.voxel_mm || data.spacing;
  viewer = {
    renderer, container, resize, frame: 0, onAutoRotate: null,
    setVisible: (i, on) => { handles[i].visible = on; },
    setAutoRotate: (on) => { controls.autoRotate = on; },
    view: (name) => { controls.autoRotate = false; viewer.onAutoRotate?.(false); flyTo(views[name]); },
    setCut: (fraction) => {
      // fraction 0 = no cut, 1 = everything removed (cut from the patient's left side).
      cutPlane.constant = fraction <= 0 ? 1e6 : (0.5 - fraction) * size.x;
    },
    setSlice: (index) => {
      if (index == null) { sliceMesh.visible = false; return; }
      sliceMesh.visible = true;
      const [cx, cz] = data.slice_plane?.center_xz || [center.x, center.z];
      sliceMesh.position.set(cx, index * voxel[0], cz);
      if (opts.sliceUrl && sliceMesh.userData.index !== index) {
        sliceMesh.userData.index = index;
        loadInto(sliceMesh.material, opts.sliceUrl(index));
      }
    },
    onHover: null, // ({name, point, dir, clientX, clientY}) in data coordinates (mm), or null
    screenshot: () => renderer.domElement.toDataURL("image/png"),
  };

  // Hover: find what is under the pointer. Tumor parts win over the possible-extent shell, which
  // wins over the organ surface (seen through it); clipped-away parts of the organ are ignored.
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let pointer = null;
  let hovered = null;
  renderer.domElement.addEventListener("pointermove", (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    pointer = { x: e.clientX, y: e.clientY, buttons: e.buttons };
  });
  renderer.domElement.addEventListener("pointerleave", () => { pointer = null; setHover(null); viewer.onHover?.(null); });
  const rank = (name) => (name === "anatomy" ? 2 : name === "possible_extent" ? 1 : 0);
  const glow = (mesh, on) => {
    if (!mesh) return;
    if (mesh.material.uniforms) mesh.material.uniforms.strength.value = on ? 1.8 : 1.0;
    else if (mesh.material.emissive) mesh.material.emissiveIntensity = on ? 3.0 : 1.0;
  };
  const setHover = (mesh) => {
    if (mesh === hovered) return;
    glow(hovered, false);
    hovered = mesh;
    glow(hovered, true);
    renderer.domElement.style.cursor = mesh ? "help" : "";
  };
  const pick = () => {
    if (!pointer || pointer.buttons) return;
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(handles.filter((h) => h.visible && h.userData.name !== "image"), false)
      .filter((h) => h.object.userData.name !== "anatomy" || cutPlane.distanceToPoint(h.point) >= 0)
      .sort((a, b) => rank(a.object.userData.name) - rank(b.object.userData.name) || a.distance - b.distance);
    const hit = hits[0];
    setHover(hit?.object || null);
    if (!viewer.onHover) return;
    if (!hit) { viewer.onHover(null); return; }
    viewer.onHover({
      name: hit.object.userData.name,
      point: group.worldToLocal(hit.point.clone()).toArray(),
      dir: raycaster.ray.direction.toArray(),
      clientX: pointer.x,
      clientY: pointer.y,
    });
  };
  let lastPick = 0;

  const tick = (now) => {
    if (now - lastPick > 60) { lastPick = now; pick(); }
    if (flight) {
      const t = Math.min(1, (now - flight.start) / flight.ms);
      const ease = 1 - Math.pow(1 - t, 3);
      camera.position.lerpVectors(flight.from, flight.to, ease);
      if (t >= 1) flight = null;
    }
    const fade = Math.min(1, (now - born) / 900);
    if (fade < 1 || handles[0]?.material.opacity === 0) {
      handles.forEach((m) => {
        if (m.material.uniforms) m.material.uniforms.strength.value = 1.0 * fade;
        else m.material.opacity = m.userData.targetOpacity * fade;
        if (fade >= 1 && !m.material.uniforms) m.material.transparent = m.userData.targetOpacity < 1;
      });
    }
    controls.update();
    renderer.render(scene, camera);
    viewer.frame = requestAnimationFrame(tick);
  };
  viewer.frame = requestAnimationFrame(tick);
  return viewer;
}

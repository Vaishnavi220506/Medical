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
  varying float vHeight;
  #include <clipping_planes_pars_vertex>
  void main() {
    vHeight = position.y;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mvPosition.xyz);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
  }`;
const FRESNEL_FRAGMENT = `
  uniform vec3 glowColor;
  uniform float strength;
  uniform float fill;
  uniform vec3 fade; // (lowest y, highest y, fade length): fade out towards both ends; length 0 = off
  varying vec3 vNormal;
  varying vec3 vView;
  varying float vHeight;
  #include <clipping_planes_pars_fragment>
  void main() {
    #include <clipping_planes_fragment>
    float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.2);
    float alpha = clamp(rim * strength + fill, 0.0, 1.0);
    if (fade.z > 0.0) {
      float y = vHeight;
      alpha *= smoothstep(min(fade.x, fade.y), min(fade.x, fade.y) + fade.z, y) * (1.0 - smoothstep(max(fade.x, fade.y) - fade.z, max(fade.x, fade.y), y));
    }
    gl_FragColor = vec4(glowColor, alpha);
  }`;

// Per sub-region look: edema is translucent so the core inside stays visible.
const STYLE = {
  edema: { opacity: 0.38, roughness: 0.5, renderOrder: 3 },
  enhancing_tumor: { opacity: 0.88, roughness: 0.28, renderOrder: 2 },
  necrotic_core: { opacity: 1.0, roughness: 0.45, renderOrder: 1 },
  nodule: { opacity: 1.0, roughness: 0.35, renderOrder: 1 },  // solid and glossy, like the brain tumor parts
  glands: { opacity: 0.4, roughness: 0.8, renderOrder: 1 },  // drawn first: the tumor shows through
  lobes: { opacity: 0.8, roughness: 0.7, renderOrder: 1 },
};

// Soft round glow texture for tumors in the illustrated body view.
function glowTexture(THREE) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.2, "rgba(255,255,255,0.85)");
  grad.addColorStop(0.45, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export async function renderMeshes(container, data) {
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
  // Some scans store the slices from head to feet (flip_vertical): mirror them so the head is up.
  const ySign = data.flip_vertical ? -1 : 1;
  const toGeometry = (m) => {
    const v = m.vertices;
    const pos = new Float32Array(v.length);
    for (let i = 0; i < v.length; i += 3) {
      pos[i] = v[i + 2];
      pos[i + 1] = ySign * v[i];
      pos[i + 2] = -v[i + 1];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    let faces = m.faces;
    if (ySign < 0) {  // a mirror turns triangles inside out: restore their winding (and normals)
      faces = faces.slice();
      for (let i = 0; i < faces.length; i += 3) [faces[i + 1], faces[i + 2]] = [faces[i + 2], faces[i + 1]];
    }
    g.setIndex(faces);
    g.computeVertexNormals();
    return g;
  };

  const group = new THREE.Group();
  const figure = data.meshes.some((m) => m.name === "body");
  const cutPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e6);
  const handles = [];
  for (const m of data.meshes) {
    const geometry = toGeometry(m);
    let mesh;
    if (m.name === "anatomy" || m.name === "body") {
      // The organ glows cyan; the illustrated body around it is a translucent blue skin that
      // fades out at the neck and below the chest.
      const body = m.name === "body";
      geometry.computeBoundingBox();
      const bb = geometry.boundingBox;
      const material = new THREE.ShaderMaterial({
        uniforms: {
          glowColor: { value: new THREE.Color(body ? 0x60a5fa : figure ? 0x7dd3fc : 0x67e8f9) },
          strength: { value: body ? 0.8 : figure ? 0.06 : 1.0 },  // breasts: shown by the body's own skin
          fill: { value: body ? 0.07 : figure ? 0.0 : 0.02 },
          fade: { value: new THREE.Vector3(bb.min.y, bb.max.y, body ? (bb.max.y - bb.min.y) * 0.16 : 0) },
        },
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
      mesh.renderOrder = body ? 11 : 10;
      mesh.userData.strength = body ? 0.8 : figure ? 0.06 : 1.0;
    } else {
      const style = STYLE[m.name] || { opacity: m.opacity ?? 0.9, roughness: 0.35, renderOrder: m.opacity < 0.5 ? 4 : 2 };
      const color = new THREE.Color(`rgb(${m.color.join(",")})`);
      const material = new THREE.MeshPhysicalMaterial({
        color,
        roughness: style.roughness,
        metalness: 0.05,
        clearcoat: m.name === "glands" ? 0 : m.name === "lobes" ? 0.3 : 0.8,
        clearcoatRoughness: 0.25,
        sheen: m.name === "lobes" ? 0 : 0.4,
        sheenColor: color.clone().offsetHSL(0, 0, 0.2),
        // In the illustrated body view the tumor glows orange, as in medical illustrations.
        // Tumors with a glow (illustrated breast view) light up orange-red.
        emissive: m.spots ? new THREE.Color(0xff3d00).multiplyScalar(0.35) : color.clone().multiplyScalar(m.name === "lobes" ? 0.08 : 0.12),
        transparent: style.opacity < 1,
        opacity: style.opacity,
        depthWrite: style.opacity >= 1,
        side: THREE.DoubleSide,
      });
      mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = style.renderOrder;
      if (m.spots) {
        // Tumor: a warm glow around each spot, as in medical illustrations.
        const texture = glowTexture(THREE);
        for (const [sl, row, col, radius] of m.spots) {
          const halo = new THREE.Sprite(new THREE.SpriteMaterial({
            map: texture, color: 0xff5500, transparent: true, opacity: 1.0,
            blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
          }));
          halo.position.set(col, ySign * sl, -row);
          const r = Math.max(radius * 7, 45);
          halo.scale.set(r, r, 1);
          halo.renderOrder = 20;
          halo.raycast = () => {};
          mesh.add(halo);
        }
      }
    }
    mesh.userData.targetOpacity = mesh.material.opacity ?? 1;
    mesh.visible = !m.hidden;
    mesh.userData.name = m.name;
    group.add(mesh);
    handles.push(mesh);
  }

  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  group.position.sub(center);
  scene.add(group);
  const radius = size.length() / 2;
  scene.fog = new THREE.Fog(0x050914, radius * 2.4, radius * 6);  // far parts fade, scaled to the scene

  // Floor grid that fades into the background.
  const grid = new THREE.PolarGridHelper(radius * 1.6, 12, 6, 64, 0x1e3a5f, 0x13233d);
  grid.position.y = -size.y / 2 - radius * 0.08;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);

  // Slice plane: where the 2D viewer currently is (a light outline, no image).
  const planeSize = data.slice_plane?.size || [size.x, size.z];
  const sliceMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(planeSize[0], planeSize[1]),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }),
  );
  sliceMesh.rotation.x = Math.PI / 2;
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
      sliceMesh.position.set(cx, ySign * index * voxel[0], cz);
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
  // Tumor parts first, then the organ area (explained like the brain: which part, how far the
  // tumor is), then the drawn milk lobes / scan glands, then the illustrated body.
  const rank = (name) => ({ body: 4, lobes: 3, glands: 3, anatomy: 2, possible_extent: 1 })[name] ?? 0;
  const glow = (mesh, on) => {
    if (!mesh) return;
    if (mesh.material.uniforms) mesh.material.uniforms.strength.value = (mesh.userData.strength ?? 1) * (on ? 1.8 : 1.0);
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
    const hits = raycaster.intersectObjects(handles.filter((h) => h.visible), false)
      .filter((h) => !["anatomy", "body"].includes(h.object.userData.name) || cutPlane.distanceToPoint(h.point) >= 0)
      .sort((a, b) => rank(a.object.userData.name) - rank(b.object.userData.name) || a.distance - b.distance);
    const hit = hits[0];
    setHover(hit?.object || null);
    if (!viewer.onHover) return;
    if (!hit) { viewer.onHover(null); return; }
    viewer.onHover({
      name: hit.object.userData.name,
      point: group.worldToLocal(hit.point.clone()).multiply(new THREE.Vector3(1, ySign, 1)).toArray(),  // data coordinates
      dir: raycaster.ray.direction.clone().multiply(new THREE.Vector3(1, ySign, 1)).toArray(),
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
        if (m.material.uniforms) m.material.uniforms.strength.value = (m.userData.strength ?? 1) * fade;
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

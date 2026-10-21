// Breast anatomy drawn around the patient's real tumor in the 3D view, in the style of a medical
// atlas: fatty tissue, milk ducts converging on the nipple, lobes of small lobules, Cooper's
// ligaments, the pectoral muscle behind the breast and the axillary lymph nodes. Everything here is
// a drawing for orientation; only the breast position and size and the tumor come from the scan.
//
// Each breast arrives as {centre: [slice, row, col] mm, radii: [up-down, front-back, sideways] mm,
// lateral: ±1}. In viewer coordinates the breast's local frame is: x = sideways (away from the
// midline), y = up, z = forward; the chest wall is at z = 0.2 * sideways radius.

// Small deterministic random numbers, so every load draws the same anatomy.
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Diagonal muscle fibres for the pectoral muscle.
function fibreTexture(THREE) {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#a8323a";
  g.fillRect(0, 0, 256, 256);
  for (let i = -256; i < 512; i += 6) {
    g.strokeStyle = i % 12 ? "rgba(255,170,170,0.22)" : "rgba(60,0,10,0.35)";
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(i, 0); g.lineTo(i + 180, 256); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function buildBreastAnatomy(THREE, shapes, { ySign = 1, cutPlane = null, reducedMotion = false } = {}) {
  const names = ["fat", "ducts", "lobules", "ligaments", "nipple", "muscle", "lymph_nodes"];
  const groups = Object.fromEntries(names.map((n) => [n, new THREE.Group()]));
  const tubes = []; // duct meshes, drawn in during the intro
  const fadeIns = []; // [material, final opacity, start s, length s]
  const clip = (planes) => [...planes, ...(cutPlane ? [cutPlane] : [])];

  const mat = {
    lobule: new THREE.MeshPhysicalMaterial({ color: 0xf2a3b3, roughness: 0.45, sheen: 0.6, sheenColor: new THREE.Color(0xffd6e0),
      emissive: 0x7a2a3a, emissiveIntensity: 0.25, transparent: true, opacity: 0.95, clippingPlanes: clip([]) }),
    duct: new THREE.MeshPhysicalMaterial({ color: 0xe0849c, roughness: 0.45, emissive: 0x5a1a2a, emissiveIntensity: 0.2,
      transparent: true, opacity: 0.95, clippingPlanes: clip([]) }),
    ligament: new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25, depthWrite: false, clippingPlanes: clip([]) }),
    nipple: new THREE.MeshPhysicalMaterial({ color: 0xc97a7a, roughness: 0.6, sheen: 0.5, sheenColor: new THREE.Color(0xffc4c4),
      transparent: true, opacity: 0.92, side: THREE.DoubleSide, clippingPlanes: clip([]) }),
    muscle: new THREE.MeshStandardMaterial({ map: fibreTexture(THREE), roughness: 0.7, transparent: true, opacity: 0.55,
      side: THREE.DoubleSide, depthWrite: false, clippingPlanes: clip([]) }),
    node: new THREE.MeshPhysicalMaterial({ color: 0x8fa68c, roughness: 0.5, sheen: 0.4, transparent: true, opacity: 0.95, clippingPlanes: clip([]) }),
    vessel: new THREE.MeshBasicMaterial({ color: 0xb9cbb5, transparent: true, opacity: 0.6, clippingPlanes: clip([]) }),
  };
  fadeIns.push([mat.lobule, mat.lobule.opacity, 1.8, 0.8], [mat.ligament, mat.ligament.opacity, 1.2, 0.8],
    [mat.nipple, mat.nipple.opacity, 0.4, 0.6], [mat.muscle, mat.muscle.opacity, 0.2, 0.8],
    [mat.node, mat.node.opacity, 2.2, 0.8], [mat.vessel, mat.vessel.opacity, 2.2, 0.8]);

  shapes.forEach((shape, b) => {
    const random = rng(1234 + b);
    const [ry, ra, rx] = shape.radii;
    const [sl, row, col] = shape.centre;
    const z0 = 0.2 * rx; // chest wall
    const front = (x, y) => ra * Math.sqrt(Math.max(1 - (x / rx) ** 2 - (y / ry) ** 2, 0)); // skin, local z
    const frame = (obj) => {
      const holder = new THREE.Group();
      holder.position.set(col, ySign * sl, -row);
      holder.scale.x = shape.lateral; // local +x points away from the midline
      holder.add(obj);
      return holder;
    };
    const tube = (points, radius, material, segments = 48) => {
      const g = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), segments, radius, 8, false);
      return new THREE.Mesh(g, material);
    };
    // Breast-local clipping: only in front of the chest wall (world plane, z = forward).
    const chestPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), row - z0);

    // Fat: a soft yellow volume with a faint lobular pattern (noise in the shader).
    const fatMat = new THREE.MeshPhysicalMaterial({ color: 0xf3d9a4, roughness: 0.85, transparent: true, opacity: 0.3,
      depthWrite: false, side: THREE.DoubleSide, clippingPlanes: clip([chestPlane]) });
    fatMat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vLocal;")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\nvLocal = position;");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", `#include <common>
          varying vec3 vLocal;
          float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
          float noise3(vec3 p) {
            vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
            return mix(mix(mix(hash3(i), hash3(i + vec3(1,0,0)), f.x), mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), f.x), f.y),
                       mix(mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), f.x), mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), f.x), f.y), f.z);
          }`)
        .replace("#include <color_fragment>", `#include <color_fragment>
          float lob = noise3(vLocal * 9.0) * 0.6 + noise3(vLocal * 19.0) * 0.4;  // fat lobules
          diffuseColor.rgb *= 0.78 + 0.35 * smoothstep(0.35, 0.75, lob);`);
    };
    fadeIns.push([fatMat, fatMat.opacity, 0.3, 0.8]);
    const fat = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 48), fatMat);
    fat.scale.set(rx * 0.96, ry * 0.96, ra * 0.96);
    groups.fat.add(frame(fat));

    // Nipple and areola at the front of the breast.
    const areola = new THREE.Mesh(new THREE.CircleGeometry(0.2 * rx, 48), mat.nipple);
    areola.position.z = front(0, 0) - 0.6;
    const nipple = new THREE.Mesh(new THREE.SphereGeometry(0.06 * rx, 24, 16), mat.nipple);
    nipple.scale.z = 1.2;
    nipple.position.z = front(0, 0);
    const nippleGroup = new THREE.Group();
    nippleGroup.add(areola, nipple);
    groups.nipple.add(frame(nippleGroup));

    // Lobes: each a few clusters of lobules at the ends of branching ducts that converge on the nipple.
    const apex = new THREE.Vector3(0, 0, front(0, 0) - 0.04 * rx);
    const lobuleSpots = [];
    const ducts = new THREE.Group();
    const lobes = 16;
    for (let i = 0; i < lobes; i++) {
      const phi = (2 * Math.PI * (i + 0.3 * random())) / lobes;
      const rho = 0.42 + 0.3 * random();
      const x = Math.cos(phi) * rho * rx, y = Math.sin(phi) * rho * ry;
      const centre = new THREE.Vector3(x, y, z0 + 0.5 * (front(x, y) - z0));
      // Main duct: gently curved (not a straight spoke), widening towards the nipple.
      const bend = (random() - 0.5) * 0.25;
      const mid = new THREE.Vector3(Math.cos(phi + bend) * rho * 0.5 * rx, Math.sin(phi + bend) * rho * 0.5 * ry, 0);
      mid.z = z0 + 0.72 * (front(mid.x, mid.y) - z0);
      const nearNipple = new THREE.Vector3(x * 0.07, y * 0.07, apex.z - 0.03 * rx);
      ducts.add(tube([centre, mid, nearNipple, apex], 0.016 * rx, mat.duct));
      for (let j = 0; j < 3; j++) { // branches to lobule clusters
        const a = phi + (j - 1) * 0.5 + (random() - 0.5) * 0.3;
        const dir = new THREE.Vector3(Math.cos(a), Math.sin(a), (random() - 0.5) * 0.8).normalize();
        const end = centre.clone().addScaledVector(dir, (0.09 + 0.06 * random()) * rx);
        end.z = Math.min(end.z, z0 + 0.85 * (front(end.x, end.y) - z0));
        ducts.add(tube([centre, centre.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 0, 0.02 * rx)), end], 0.009 * rx, mat.duct, 16));
        for (let k = 0; k < 7; k++) { // a grape-like cluster of lobules
          const p = end.clone().add(new THREE.Vector3(random() - 0.5, random() - 0.5, random() - 0.5).multiplyScalar(0.09 * rx));
          lobuleSpots.push([p, (0.03 + 0.025 * random()) * rx]);
        }
      }
    }
    ducts.traverse((o) => { if (o.isMesh) tubes.push(o); });
    groups.ducts.add(frame(ducts));
    const lobuleMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10), mat.lobule, lobuleSpots.length);
    const m4 = new THREE.Matrix4();
    lobuleSpots.forEach(([p, r], i) => lobuleMesh.setMatrixAt(i, m4.compose(p, new THREE.Quaternion(), new THREE.Vector3(r, r, r))));
    groups.lobules.add(frame(lobuleMesh));

    // Cooper's ligaments: thin strands from the chest wall out to the skin.
    const ligaments = new THREE.Group();
    for (let i = 0; i < 12; i++) {
      const phi = 2 * Math.PI * (i + random()) / 12;
      const r0 = 0.3 + 0.5 * random();
      const a = new THREE.Vector3(Math.cos(phi) * r0 * rx, Math.sin(phi) * r0 * ry, z0);
      const r1 = Math.min(r0 + 0.15, 0.95);
      const bx = Math.cos(phi + 0.1) * r1 * rx, by = Math.sin(phi + 0.1) * r1 * ry;
      const bEnd = new THREE.Vector3(bx, by, z0 + 0.92 * (front(bx, by) - z0));
      const m = a.clone().lerp(bEnd, 0.5); m.z += 0.05 * rx;
      ligaments.add(tube([a, m, bEnd], 0.006 * rx, mat.ligament, 20));
    }
    groups.ligaments.add(frame(ligaments));

    // Pectoral muscle: a fan behind the breast, from the breastbone towards the armpit.
    const fan = new THREE.Shape();
    fan.moveTo(-0.95 * rx, -0.8 * ry);
    fan.quadraticCurveTo(0.3 * rx, -1.0 * ry, 1.25 * rx, 0.7 * ry);
    fan.lineTo(1.32 * rx, 0.95 * ry);
    fan.quadraticCurveTo(0.2 * rx, 1.25 * ry, -0.95 * rx, 0.95 * ry);
    fan.closePath();
    const slab = new THREE.ShapeGeometry(fan, 24);
    const pos = slab.attributes.position, uv = slab.attributes.uv;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      pos.setZ(i, -0.15 * (x / rx) ** 2 * rx);  // follows the curve of the chest
      uv.setXY(i, x / (2.6 * rx) + 0.5, y / (2.4 * ry) + 0.5);
    }
    slab.computeVertexNormals();
    const muscle = new THREE.Mesh(slab, mat.muscle);
    muscle.position.z = z0 - 0.1 * rx;
    groups.muscle.add(frame(muscle));

    // Axillary lymph nodes: a chain from the upper outer breast towards the armpit.
    const chain = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.75 * rx, 0.45 * ry, z0 + 0.12 * ra),
      new THREE.Vector3(1.05 * rx, 0.75 * ry, z0 + 0.02 * rx),
      new THREE.Vector3(1.35 * rx, 1.0 * ry, z0 - 0.06 * rx),
    ]);
    const nodes = new THREE.Group();
    nodes.add(new THREE.Mesh(new THREE.TubeGeometry(chain, 40, 0.008 * rx, 6, false), mat.vessel));
    for (let i = 0; i < 6; i++) {
      const node = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), mat.node);
      const r = (0.045 + 0.025 * random()) * rx;
      node.scale.set(r, r * 0.7, r * 0.7);
      node.position.copy(chain.getPoint(i / 5));
      node.rotation.z = random();
      nodes.add(node);
    }
    groups.lymph_nodes.add(frame(nodes));
  });

  for (const [name, group] of Object.entries(groups)) {
    group.userData.name = name;
    group.traverse((o) => { o.userData.name = name; });
  }
  const counts = tubes.map((t) => t.geometry.index.count);
  const update = (seconds) => {
    // Intro: tissue fades in, the ducts draw in towards the nipple, then the lobules appear.
    const t = reducedMotion ? 1e3 : seconds;
    const duct = Math.min(1, Math.max(0, (t - 0.6) / 1.4));
    tubes.forEach((m, i) => m.geometry.setDrawRange(0, Math.floor(counts[i] * duct / 3) * 3));
    for (const [material, final, start, length] of fadeIns) material.opacity = final * Math.min(1, Math.max(0, (t - start) / length));
    return t > 3.2; // intro finished
  };
  update(0);
  return { groups, update };
}

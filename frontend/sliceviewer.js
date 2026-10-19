// Interactive slice viewer: zoom (wheel / pinch / buttons), pan (drag), brightness &
// contrast, overlay opacity, a ruler that measures in mm, and fullscreen.
//
// The server renders two PNGs per slice: the plain scan ("original") and the chosen
// layer (overlay / heatmap / uncertainty) already blended onto it. Drawing the layer
// over the plain scan with partial alpha gives an adjustable layer opacity.

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;

export class SliceViewer {
  constructor(container, { onSliceStep, onZoom, onHover } = {}) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "slice-canvas";
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", "Scan viewer: scroll to zoom, drag to pan, Shift+scroll or arrow keys to change slice");
    container.replaceChildren(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.onSliceStep = onSliceStep || (() => {});
    // Called with ([x, y] image pixel, mouse event) while the pointer is over the image, null when it leaves.
    this.onHover = onHover || (() => {});
    this.onZoom = onZoom || (() => {});

    this.base = null; // HTMLImageElement of the plain scan
    this.layer = null; // HTMLImageElement of the blended layer (or null)
    this.opacity = 1;
    this.brightness = 1;
    this.contrast = 1;
    this.spacing = null; // [row mm, col mm]
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.fitted = false;
    this.userMoved = false; // true once the user zooms or pans; resizes then keep their view
    this.measuring = false;
    this.lines = new Map(); // slice key -> [{a:[x,y], b:[x,y]}]
    this.sliceKey = "";
    this.pointers = new Map();
    this.drag = null;
    this.hover = null;
    this._loadToken = 0;

    this._bind();
    new ResizeObserver(() => this._resize()).observe(container);
    this._resize();
  }

  // ------------------------------------------------------------------ public API
  async show(baseUrl, layerUrl, sliceKey) {
    const token = ++this._loadToken;
    const [base, layer] = await Promise.all([loadImage(baseUrl), layerUrl ? loadImage(layerUrl) : null]);
    if (token !== this._loadToken) return; // a newer slice was requested meanwhile
    const sizeChanged = !this.base || this.base.naturalWidth !== base.naturalWidth || this.base.naturalHeight !== base.naturalHeight;
    this.base = base;
    this.layer = layer;
    this.sliceKey = sliceKey;
    if (sizeChanged || !this.fitted) this.fit();
    this.draw();
  }

  setSpacing(rowMm, colMm) {
    this.spacing = rowMm && colMm ? [rowMm, colMm] : null;
  }

  setDisplay({ opacity, brightness, contrast }) {
    if (opacity !== undefined) this.opacity = opacity;
    if (brightness !== undefined) this.brightness = brightness;
    if (contrast !== undefined) this.contrast = contrast;
    this.draw();
  }

  fit() {
    if (!this.base) return;
    const { w, h } = this._size();
    const s = Math.min(w / this.base.naturalWidth, h / this.base.naturalHeight);
    this.scale = s;
    this.tx = (w - this.base.naturalWidth * s) / 2;
    this.ty = (h - this.base.naturalHeight * s) / 2;
    this.fitted = true;
    this.userMoved = false;
    this.onZoom(this.zoomPercent());
    this.draw();
  }

  zoomBy(factor, cx, cy) {
    if (!this.base) return;
    const { w, h } = this._size();
    cx = cx ?? w / 2;
    cy = cy ?? h / 2;
    const fitScale = Math.min(w / this.base.naturalWidth, h / this.base.naturalHeight);
    const next = Math.min(Math.max(this.scale * factor, fitScale * MIN_ZOOM), fitScale * MAX_ZOOM);
    const k = next / this.scale;
    // Keep the point under the cursor fixed while zooming.
    this.tx = cx - (cx - this.tx) * k;
    this.ty = cy - (cy - this.ty) * k;
    this.scale = next;
    this.userMoved = true;
    this.onZoom(this.zoomPercent());
    this.draw();
  }

  zoomPercent() {
    if (!this.base) return 100;
    const { w, h } = this._size();
    return Math.round((100 * this.scale) / Math.min(w / this.base.naturalWidth, h / this.base.naturalHeight));
  }

  setMeasuring(on) {
    this.measuring = on;
    this.canvas.classList.toggle("measuring", on);
  }

  clearMeasurements() {
    this.lines.delete(this.sliceKey);
    this.draw();
  }

  toggleFullscreen(element) {
    if (document.fullscreenElement) document.exitFullscreen();
    else element.requestFullscreen?.();
  }

  // ------------------------------------------------------------------ drawing
  draw() {
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!this.base) return;

    ctx.setTransform(dpr * this.scale, 0, 0, dpr * this.scale, dpr * this.tx, dpr * this.ty);
    // Show crisp pixels when zoomed in a lot, smooth otherwise.
    ctx.imageSmoothingEnabled = this.zoomPercent() < 300;
    ctx.filter = `brightness(${this.brightness}) contrast(${this.contrast})`;
    ctx.drawImage(this.base, 0, 0);
    if (this.layer) {
      ctx.globalAlpha = this.opacity;
      ctx.drawImage(this.layer, 0, 0);
      ctx.globalAlpha = 1;
    }
    ctx.filter = "none";

    // Ruler lines (image coordinates), labels in screen space.
    const lines = [...(this.lines.get(this.sliceKey) || []), ...(this.drag?.line ? [this.drag.line] : [])];
    for (const line of lines) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const [ax, ay] = this._toScreen(line.a);
      const [bx, by] = this._toScreen(line.b);
      ctx.strokeStyle = "#22d3ee";
      ctx.fillStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      for (const [x, y] of [[ax, ay], [bx, by]]) {
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      const text = this._lengthText(line);
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      const tw = ctx.measureText(text).width;
      const lx = (ax + bx) / 2 + 8;
      const ly = (ay + by) / 2 - 8;
      ctx.fillStyle = "rgba(0,0,0,.75)";
      ctx.fillRect(lx - 4, ly - 13, tw + 8, 18);
      ctx.fillStyle = "#e0f2fe";
      ctx.fillText(text, lx, ly);
    }

    // Cursor readout in the corner.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const info = [`${this.zoomPercent()}%`];
    if (this.hover) info.push(`x ${this.hover[0]}  y ${this.hover[1]}`);
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillStyle = "rgba(229,233,242,.75)";
    ctx.fillText(info.join("   "), 8, this._size().h - 8);
  }

  // ------------------------------------------------------------------ helpers
  _size() {
    const r = this.container.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const { w, h } = this._size();
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    if (this.userMoved) this.draw();
    else this.fit();
  }

  _toImage(sx, sy) {
    return [(sx - this.tx) / this.scale, (sy - this.ty) / this.scale];
  }

  _toScreen([ix, iy]) {
    return [ix * this.scale + this.tx, iy * this.scale + this.ty];
  }

  _local(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _lengthText(line) {
    const dx = line.b[0] - line.a[0];
    const dy = line.b[1] - line.a[1];
    if (this.spacing) {
      const mm = Math.hypot(dx * this.spacing[1], dy * this.spacing[0]);
      return `${mm.toFixed(1)} mm`;
    }
    return `${Math.hypot(dx, dy).toFixed(0)} px`;
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (e.shiftKey) {
        this.onSliceStep(Math.sign(e.deltaY || e.deltaX));
        return;
      }
      const [x, y] = this._local(e);
      this.zoomBy(Math.pow(1.0015, -e.deltaY), x, y);
    }, { passive: false });

    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      c.focus();
      this.pointers.set(e.pointerId, this._local(e));
      if (this.pointers.size === 2) {
        const [p1, p2] = [...this.pointers.values()];
        this.drag = { pinch: Math.hypot(p1[0] - p2[0], p1[1] - p2[1]) };
      } else if (this.measuring && this.base) {
        const p = this._toImage(...this._local(e));
        this.drag = { line: { a: p, b: p } };
      } else {
        this.drag = { last: this._local(e) };
        c.classList.add("dragging");
      }
    });

    c.addEventListener("pointermove", (e) => {
      const pos = this._local(e);
      if (this.base) {
        const [ix, iy] = this._toImage(...pos);
        const inside = ix >= 0 && iy >= 0 && ix < this.base.naturalWidth && iy < this.base.naturalHeight;
        this.hover = inside ? [Math.floor(ix), Math.floor(iy)] : null;
        if (!this.pointers.size) this.onHover(this.hover, e);
      }
      if (!this.pointers.has(e.pointerId)) {
        this.draw();
        return;
      }
      this.pointers.set(e.pointerId, pos);
      if (this.drag?.pinch && this.pointers.size === 2) {
        const [p1, p2] = [...this.pointers.values()];
        const d = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
        this.zoomBy(d / this.drag.pinch, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2);
        this.drag.pinch = d;
      } else if (this.drag?.line) {
        this.drag.line.b = this._toImage(...pos);
        this.draw();
      } else if (this.drag?.last) {
        this.tx += pos[0] - this.drag.last[0];
        this.ty += pos[1] - this.drag.last[1];
        this.userMoved = true;
        this.drag.last = pos;
        this.draw();
      }
    });

    const end = (e) => {
      this.pointers.delete(e.pointerId);
      if (this.drag?.line) {
        const { a, b } = this.drag.line;
        if (Math.hypot(b[0] - a[0], b[1] - a[1]) * this.scale > 4) {
          const list = this.lines.get(this.sliceKey) || [];
          list.push(this.drag.line);
          this.lines.set(this.sliceKey, list);
        }
      }
      if (this.pointers.size === 0) {
        this.drag = null;
        c.classList.remove("dragging");
      }
      this.draw();
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("pointerleave", () => { this.hover = null; this.onHover(null); this.draw(); });
    c.addEventListener("dblclick", () => this.fit());

    c.addEventListener("keydown", (e) => {
      const actions = {
        "+": () => this.zoomBy(1.25), "=": () => this.zoomBy(1.25), "-": () => this.zoomBy(0.8), "0": () => this.fit(),
        ArrowUp: () => this.onSliceStep(1), ArrowDown: () => this.onSliceStep(-1),
        ArrowRight: () => this.onSliceStep(1), ArrowLeft: () => this.onSliceStep(-1),
      };
      if (actions[e.key]) {
        e.preventDefault();
        actions[e.key]();
      }
    });
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${url}`));
    img.src = url;
  });
}

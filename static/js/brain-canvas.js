/**
 * DecodeLabs — Project 2: AI Data Classification
 * Interactive 3D Golden Brain Canvas (Brainscape Explorer Port)
 */

(function () {
  function rand(min, max) {
    return min + Math.random() * (max - min);
  }

  /** Sample a point on a brain-like surface (two folded hemispheres + cerebellum + stem). */
  function brainPoint() {
    const roll = Math.random();

    // cerebellum
    if (roll < 0.11) {
      const u = Math.random() * 2 - 1;
      const th = Math.random() * Math.PI * 2;
      const r = Math.sqrt(1 - u * u);
      return {
        x: r * Math.cos(th) * 0.34,
        y: -0.44 + u * 0.17,
        z: -0.72 + r * Math.sin(th) * 0.22,
      };
    }

    // brain stem
    if (roll < 0.15) {
      const t = Math.random();
      const th = Math.random() * Math.PI * 2;
      return {
        x: Math.cos(th) * 0.09,
        y: -0.42 - t * 0.36,
        z: -0.34 + t * 0.16 + Math.sin(th) * 0.09,
      };
    }

    // cerebrum shell
    const u = Math.random() * 2 - 1;
    const th = Math.random() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    let x = r * Math.cos(th) * 0.86;
    let y = u * 0.74;
    let z = r * Math.sin(th) * 1.12;

    if (y < -0.2) y = -0.2 + (y + 0.2) * 0.5;
    z += 0.1 * Math.sin(y * 2.4);
    x += (x >= 0 ? 1 : -1) * 0.05;

    const fold =
      0.05 * Math.sin(x * 10 + z * 8) +
      0.042 * Math.sin(y * 12 - z * 7) +
      0.03 * Math.sin(z * 14 + x * 6);
    const n = Math.hypot(x, y, z) || 1;
    return { x: x + (x / n) * fold, y: y + (y / n) * fold, z: z + (z / n) * fold };
  }

  function initBrainCanvas() {
    const canvas = document.getElementById("brainCanvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = 0;
    let height = 0;

    const resize = () => {
      const parent = canvas.parentElement || canvas;
      const rect = parent.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width || 400;
      height = rect.height || 400;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = width + "px";
      canvas.style.height = height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();

    // Wireframe nodes
    const NODES = 650;
    const nodes = Array.from({ length: NODES }, () => ({
      ...brainPoint(),
      s: rand(0.8, 1.9),
    }));

    // Dust for volume
    const dust = Array.from({ length: 500 }, () => {
      const p = brainPoint();
      const k = rand(0.82, 0.99);
      return { x: p.x * k, y: p.y * k, z: p.z * k, s: rand(0.3, 0.8) };
    });

    // Mesh edges: each node links to its 3 nearest neighbors
    const edges = [];
    const seen = new Set();
    for (let i = 0; i < NODES; i++) {
      const a = nodes[i];
      const best = [];
      for (let j = 0; j < NODES; j++) {
        if (j === i) continue;
        const b = nodes[j];
        const d = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2;
        if (best.length < 3) best.push({ j, d });
        else {
          let worst = 0;
          for (let k = 1; k < best.length; k++) {
            if (best[k].d > best[worst].d) worst = k;
          }
          if (d < best[worst].d) best[worst] = { j, d };
        }
      }
      for (const b of best) {
        const key = i < b.j ? `${i}_${b.j}` : `${b.j}_${i}`;
        if (seen.has(key)) continue;
        seen.add(key);
        edges.push([i, b.j]);
      }
    }

    // Axons growing out of surface
    const axons = Array.from({ length: 30 }, () => {
      const base = brainPoint();
      const n = Math.hypot(base.x, base.y, base.z) || 1;
      let d = { x: base.x / n, y: base.y / n, z: base.z / n };
      const len = rand(0.22, 0.5);
      const segs = [{ ...base }];
      let cur = { ...base };
      for (let i = 0; i < 4; i++) {
        d = { x: d.x + rand(-0.2, 0.2), y: d.y + rand(-0.2, 0.2), z: d.z + rand(-0.2, 0.2) };
        const dn = Math.hypot(d.x, d.y, d.z) || 1;
        d = { x: d.x / dn, y: d.y / dn, z: d.z / dn };
        cur = {
          x: cur.x + (d.x * len) / 4,
          y: cur.y + (d.y * len) / 4,
          z: cur.z + (d.z * len) / 4,
        };
        segs.push({ ...cur });
      }
      return { segs, speed: rand(0.25, 0.6), phase: Math.random() };
    });

    let raf = 0;
    let t = 0;
    let angle = 0;
    let last = 0;

    const project = (p, cosA, sinA, tilt, scale) => {
      const x = p.x * cosA + p.z * sinA;
      let z = -p.x * sinA + p.z * cosA;
      const ct = Math.cos(tilt);
      const st = Math.sin(tilt);
      const y = p.y * ct - z * st;
      z = p.y * st + z * ct;
      const persp = 3.4 / (3.4 + z);
      return {
        sx: width / 2 + x * scale * persp,
        sy: height / 2 + y * scale * persp,
        depth: z,
        persp,
      };
    };

    const nodeProj = nodes.map(() => ({ sx: 0, sy: 0, depth: 0, persp: 1 }));
    const BUCKETS = 5;

    const frame = (now) => {
      const dt = last ? Math.min((now - last) / 1000, 0.05) : 0.016;
      last = now;
      t += dt;
      angle += dt * 0.5;
      const tilt = -0.16 + Math.sin(t * 0.25) * 0.05;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const scale = Math.min(width, height) * 0.38;

      ctx.clearRect(0, 0, width, height);
      ctx.lineCap = "round";

      const shadeOf = (depth) => Math.max(0.1, 1 - (depth + 1.3) / 2.9);

      for (let i = 0; i < nodes.length; i++) {
        const pr = project(nodes[i], cosA, sinA, tilt, scale);
        const dst = nodeProj[i];
        dst.sx = pr.sx;
        dst.sy = pr.sy;
        dst.depth = pr.depth;
        dst.persp = pr.persp;
      }

      // Glowing mesh
      ctx.shadowColor = "rgba(234, 179, 8, 0.95)";
      ctx.shadowBlur = 8;
      for (let b = 0; b < BUCKETS; b++) {
        const sh = (b + 0.5) / BUCKETS;
        ctx.strokeStyle = `rgba(234, ${Math.round(179 + 40 * sh)}, ${Math.round(8 + 40 * sh)}, ${0.2 + 0.6 * sh})`;
        ctx.lineWidth = 0.6 + 1.0 * sh;
        ctx.beginPath();
        let any = false;
        for (const [ai, bi] of edges) {
          const a = nodeProj[ai];
          const c = nodeProj[bi];
          const s = shadeOf((a.depth + c.depth) / 2);
          if (Math.min(BUCKETS - 1, Math.floor(s * BUCKETS)) !== b) continue;
          ctx.moveTo(a.sx, a.sy);
          ctx.lineTo(c.sx, c.sy);
          any = true;
        }
        if (any) ctx.stroke();
      }

      // Axons + travelling pulses
      for (const ax of axons) {
        const pts = ax.segs.map((p) => project(p, cosA, sinA, tilt, scale));
        const sh = shadeOf(pts[pts.length - 1].depth);
        ctx.strokeStyle = `rgba(250, 204, 21, ${0.3 + 0.5 * sh})`;
        ctx.lineWidth = 1.0 * sh + 0.4;
        ctx.beginPath();
        ctx.moveTo(pts[0].sx, pts[0].sy);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].sx, pts[i].sy);
        ctx.stroke();

        const tip = pts[pts.length - 1];
        ctx.fillStyle = `rgba(254, 240, 138, ${0.6 + 0.4 * sh})`;
        ctx.beginPath();
        ctx.arc(tip.sx, tip.sy, 2.0 * tip.persp, 0, Math.PI * 2);
        ctx.fill();

        const prog = (t * ax.speed + ax.phase) % 1;
        const seg = Math.min(pts.length - 2, Math.floor(prog * (pts.length - 1)));
        const local = prog * (pts.length - 1) - seg;
        const a = pts[seg];
        const b = pts[seg + 1];
        const px = a.sx + (b.sx - a.sx) * local;
        const py = a.sy + (b.sy - a.sy) * local;
        ctx.fillStyle = `rgba(255, 255, 255, ${0.7 + 0.3 * sh})`;
        ctx.beginPath();
        ctx.arc(px, py, 2.6 * a.persp, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;

      // Nodes
      for (let i = 0; i < nodes.length; i++) {
        const p = nodes[i];
        const pr = nodeProj[i];

        const sh = shadeOf(pr.depth);
        const flick = 0.8 + 0.2 * Math.sin(t * 2.4 + i);
        ctx.fillStyle = `rgba(234, ${Math.round(179 + 30 * sh)}, ${Math.round(8 + 50 * sh)}, ${0.3 + 0.7 * sh})`;
        ctx.beginPath();
        ctx.arc(pr.sx, pr.sy, Math.max(0.6, p.s * pr.persp * flick * 0.9), 0, Math.PI * 2);
        ctx.fill();
      }

      // Interior dust
      for (const p of dust) {
        const pr = project(p, cosA, sinA, tilt, scale);
        const sh = shadeOf(pr.depth);
        ctx.fillStyle = `rgba(250, 204, 21, ${0.15 + 0.4 * sh})`;
        ctx.beginPath();
        ctx.arc(pr.sx, pr.sy, Math.max(0.4, p.s * pr.persp), 0, Math.PI * 2);
        ctx.fill();
      }

      // Glowing core chip
      const core = project({ x: 0, y: 0.02, z: 0 }, cosA, sinA, tilt, scale);
      const chip = 0.05 * scale * core.persp * (1 + 0.04 * Math.sin(t * 2));
      const cg = ctx.createRadialGradient(core.sx, core.sy, 0, core.sx, core.sy, chip * 6);
      cg.addColorStop(0, "rgba(254, 240, 138, 0.9)");
      cg.addColorStop(0.3, "rgba(234, 179, 8, 0.5)");
      cg.addColorStop(1, "rgba(202, 138, 4, 0)");
      ctx.fillStyle = cg;
      ctx.beginPath();
      ctx.arc(core.sx, core.sy, chip * 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.save();
      ctx.translate(core.sx, core.sy);
      ctx.shadowColor = "rgba(234, 179, 8, 1)";
      ctx.shadowBlur = 20;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(-chip, -chip, chip * 2, chip * 2);
      ctx.strokeRect(-chip * 0.5, -chip * 0.5, chip, chip);
      ctx.restore();
      ctx.shadowBlur = 0;

      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);

    window.addEventListener("resize", resize);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initBrainCanvas);
  } else {
    initBrainCanvas();
  }
})();

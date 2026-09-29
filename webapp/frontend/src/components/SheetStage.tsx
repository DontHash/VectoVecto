import { createSignal, onCleanup, onMount, Show } from "solid-js";

/**
 * A calm "desk": two real pipeline pages as sheets in a shallow 3D space.
 * Pointer parallax + a slow idle drift, nothing else. Falls back to a static
 * pair of images for prefers-reduced-motion or when WebGL is unavailable.
 *
 * three.js is imported lazily so pages that never show the stage don't pay
 * for it.
 */
export function SheetStage(props: { pages: [string, string]; alts: [string, string] }) {
  const [fallback, setFallback] = createSignal(false);
  let host: HTMLDivElement | undefined;

  onMount(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setFallback(true);
      return;
    }

    let disposed = false;
    const disposables: Array<{ dispose: () => void }> = [];
    let cleanup: (() => void) | undefined;

    void (async () => {
      let THREE: typeof import("three");
      try {
        THREE = await import("three");
      } catch {
        setFallback(true);
        return;
      }
      if (disposed) return;

      let renderer: import("three").WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      } catch {
        setFallback(true);
        return;
      }
      if (!renderer.getContext()) {
        setFallback(true);
        return;
      }

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 60);
      camera.position.set(0, 0, 7.15);

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      host!.appendChild(renderer.domElement);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      renderer.domElement.setAttribute("aria-hidden", "true");

      // window light — soft, no drama
      scene.add(new THREE.HemisphereLight(0xffffff, 0xd9d2c2, 1.05));
      const key = new THREE.DirectionalLight(0xfff6e6, 1.1);
      key.position.set(2.5, 3.5, 4);
      scene.add(key);

      const group = new THREE.Group();
      scene.add(group);

      const loader = new THREE.TextureLoader();

      const load = (url: string) =>
        new Promise<import("three").Texture | null>((resolve) => {
          loader.load(
            url,
            (tex) => {
              tex.colorSpace = THREE.SRGBColorSpace;
              tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
              disposables.push(tex);
              resolve(tex);
            },
            undefined,
            () => resolve(null),
          );
        });

      const makeSheet = (tex: import("three").Texture | null, w: number, h: number) => {
        const geo = new THREE.PlaneGeometry(w, h);
        const mat = new THREE.MeshStandardMaterial({
          map: tex,
          roughness: 1,
          metalness: 0,
          transparent: true,
        });
        disposables.push(geo, mat);
        return new THREE.Mesh(geo, mat);
      };

      // geometry from the real page ratios (1500x2444 and 1200x1697)
      const letterpress = makeSheet(null, 2.24, 3.64);
      letterpress.rotation.set(0, 0.34, 0.012);
      letterpress.position.set(-1.12, 0.22, -0.15);
      group.add(letterpress);

      const invoice = makeSheet(null, 1.86, 2.63);
      invoice.rotation.set(0, -0.24, -0.02);
      invoice.position.set(1.14, 0.3, 0.55);
      group.add(invoice);

      void (async () => {
        const [texA, texB] = await Promise.all([load(props.pages[0]), load(props.pages[1])]);
        if (disposed) return;
        if (texA) (letterpress.material as import("three").MeshStandardMaterial).map = texA;
        if (texB) (invoice.material as import("three").MeshStandardMaterial).map = texB;
        (letterpress.material as import("three").MeshStandardMaterial).needsUpdate = true;
        (invoice.material as import("three").MeshStandardMaterial).needsUpdate = true;
      })();

      // pointer parallax
      let targetX = 0;
      let targetY = 0;
      const onPointer = (e: PointerEvent) => {
        const w = window.innerWidth || 1;
        const h = window.innerHeight || 1;
        targetX = (e.clientX / w) * 2 - 1;
        targetY = (e.clientY / h) * 2 - 1;
      };
      window.addEventListener("pointermove", onPointer, { passive: true });

      const resize = () => {
        if (!host) return;
        const w = host.clientWidth || 1;
        const h = host.clientHeight || 1;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      const ro = new ResizeObserver(resize);
      ro.observe(host!);
      resize();

      let visible = true;
      const io = new IntersectionObserver(
        (entries) => {
          visible = entries.some((entry) => entry.isIntersecting);
        },
        { threshold: 0.01 },
      );
      io.observe(host!);

      let raf = 0;
      const clock = new THREE.Clock();
      const tick = () => {
        raf = window.requestAnimationFrame(tick);
        if (!visible || document.hidden) return;
        const t = clock.getElapsedTime();
        group.rotation.y += (targetX * 0.1 - group.rotation.y) * 0.05;
        group.rotation.x += (-targetY * 0.05 - group.rotation.x) * 0.05;
        group.position.x += (targetX * 0.07 - group.position.x) * 0.05;
        group.position.y = Math.sin(t * 0.45) * 0.035;
        renderer.render(scene, camera);
      };
      tick();

      cleanup = () => {
        window.removeEventListener("pointermove", onPointer);
        window.cancelAnimationFrame(raf);
        ro.disconnect();
        io.disconnect();
        for (const d of disposables) d.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();

    onCleanup(() => {
      disposed = true;
      cleanup?.();
    });
  });

  return (
    <Show
      when={!fallback()}
      fallback={
        <div class="sheet-stage__fallback">
          <img src={props.pages[0]} alt={props.alts[0]} />
          <img src={props.pages[1]} alt={props.alts[1]} />
        </div>
      }
    >
      <div ref={host} class="sheet-stage__canvas" />
    </Show>
  );
}

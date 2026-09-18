// Pan and zoom for mermaid diagrams, the way a map embed does it:
// visible + / − / reset buttons, drag to pan, Ctrl + scroll (or a trackpad
// pinch) to zoom at the cursor, double-click to reset. A plain scroll still
// scrolls the page, except in the fullscreen viewer, where it zooms.
(() => {
    const MIN_SCALE = 1;
    const MAX_SCALE = 8;
    const DRAG_THRESHOLD_PX = 4;

    const attached = new WeakSet();

    const attach = (svg) => {
        if (attached.has(svg) || !svg.viewBox.baseVal.width) return;
        attached.add(svg);

        const host = svg.parentElement;
        const fullscreen = !!svg.closest(".mermaid-fullscreen-modal");
        // The fullscreen viewer clones the diagram as it is — zoomed, and with
        // buttons that lost their handlers. Start the clone from the full view.
        host.querySelectorAll(".mermaid-zoom-bar, .mermaid-zoom-hint").forEach((el) => el.remove());
        if (!svg.dataset.homeViewBox) svg.dataset.homeViewBox = svg.getAttribute("viewBox");
        const [x, y, w, h] = svg.dataset.homeViewBox.split(/[\s,]+/).map(Number);
        const home = { x, y, w, h };
        let view = { ...home };
        svg.setAttribute("viewBox", svg.dataset.homeViewBox);

        const apply = () => {
            svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
            svg.classList.toggle("is-zoomed", view.w < home.w);
        };

        const clamp = () => {
            view.x = Math.min(Math.max(view.x, home.x), home.x + home.w - view.w);
            view.y = Math.min(Math.max(view.y, home.y), home.y + home.h - view.h);
        };

        // Client pixels -> diagram coordinates, letterboxing included.
        const toDiagram = (clientX, clientY) => {
            const p = new DOMPoint(clientX, clientY).matrixTransform(svg.getScreenCTM().inverse());
            return { x: p.x, y: p.y };
        };

        const zoomAt = (factor, clientX, clientY) => {
            const scale = Math.min(Math.max((home.w / view.w) * factor, MIN_SCALE), MAX_SCALE);
            const at = toDiagram(clientX, clientY);
            const w = home.w / scale;
            const h = home.h / scale;
            view = {
                x: at.x - ((at.x - view.x) * w) / view.w,
                y: at.y - ((at.y - view.y) * h) / view.h,
                w,
                h,
            };
            clamp();
            apply();
        };

        const zoomCentre = (factor) => {
            const r = svg.getBoundingClientRect();
            zoomAt(factor, r.left + r.width / 2, r.top + r.height / 2);
        };

        const reset = () => {
            view = { ...home };
            apply();
        };

        // --- buttons ---------------------------------------------------------
        const bar = document.createElement("div");
        bar.className = "mermaid-zoom-bar";
        const button = (label, title, onClick) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            b.title = title;
            b.setAttribute("aria-label", title);
            b.addEventListener("click", onClick);
            bar.appendChild(b);
        };
        button("+", "Zoom in", () => zoomCentre(1.4));
        button("−", "Zoom out", () => zoomCentre(1 / 1.4));
        button("⟲", "Reset zoom (or double-click the diagram)", reset);

        const hint = document.createElement("div");
        hint.className = "mermaid-zoom-hint";
        hint.textContent = "Ctrl + scroll to zoom · drag to pan";

        host.classList.add("mermaid-zoom-host");
        host.append(bar, hint);

        // --- wheel -----------------------------------------------------------
        let hintTimer = null;
        svg.addEventListener(
            "wheel",
            (e) => {
                if (!fullscreen && !e.ctrlKey && !e.metaKey) {
                    hint.classList.add("visible");
                    clearTimeout(hintTimer);
                    hintTimer = setTimeout(() => hint.classList.remove("visible"), 1500);
                    return; // the page scrolls
                }
                e.preventDefault();
                zoomAt(Math.exp(-e.deltaY * (e.ctrlKey && Math.abs(e.deltaY) < 20 ? 0.02 : 0.002)), e.clientX, e.clientY);
            },
            { passive: false },
        );

        // --- drag ------------------------------------------------------------
        let drag = null;
        svg.addEventListener("pointerdown", (e) => {
            if (e.button !== 0 || view.w >= home.w) return;
            drag = { x: e.clientX, y: e.clientY, view: { ...view }, moved: false };
        });
        svg.addEventListener("pointermove", (e) => {
            if (!drag) return;
            const dx = e.clientX - drag.x;
            const dy = e.clientY - drag.y;
            if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
            if (!drag.moved) {
                drag.moved = true;
                svg.setPointerCapture(e.pointerId);
                svg.classList.add("is-dragging");
            }
            const perPixel = 1 / svg.getScreenCTM().a;
            view.x = drag.view.x - dx * perPixel;
            view.y = drag.view.y - dy * perPixel;
            clamp();
            apply();
        });
        const endDrag = () => {
            if (!drag) return;
            const moved = drag.moved;
            drag = null;
            svg.classList.remove("is-dragging");
            if (moved) {
                // A drag that ends on a box must not follow the box's link.
                const swallow = (c) => (c.preventDefault(), c.stopPropagation());
                svg.addEventListener("click", swallow, { capture: true, once: true });
                setTimeout(() => svg.removeEventListener("click", swallow, { capture: true }), 0);
            }
        };
        svg.addEventListener("pointerup", endDrag);
        svg.addEventListener("pointercancel", endDrag);

        svg.addEventListener("dblclick", (e) => {
            e.preventDefault();
            reset();
        });
    };

    const scan = () => document.querySelectorAll(".mermaid > svg").forEach(attach);

    // Diagrams render asynchronously, again on a theme switch, and are cloned
    // into the fullscreen viewer — each of those is a new <svg>.
    new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
    document.addEventListener("DOMContentLoaded", scan);
})();

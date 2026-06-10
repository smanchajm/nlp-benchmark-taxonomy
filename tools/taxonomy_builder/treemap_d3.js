(function () {
  "use strict";

  function truncate(text, max) {
    const raw = String(text || "");
    if (raw.length <= max) {
      return raw;
    }
    return `${raw.slice(0, Math.max(1, max - 1))}…`;
  }

  function buildHierarchy(node, options) {
    const children = (node.children || []).map((child) => buildHierarchy(child, options));
    const papers = Math.max(0, Number(options.nodePaperCount(node.id) || 0));
    return {
      id: node.id,
      label: node.label,
      description: node.description || "",
      value: Math.max(1, papers),
      children,
      papers,
    };
  }

  function render(container, options) {
    container.innerHTML = "";
    if (typeof window.d3 === "undefined") {
      container.innerHTML = "<div>D3 introuvable pour treemap.</div>";
      return;
    }
    if (!options || !options.root) {
      container.innerHTML = "<div>Aucune racine pour treemap.</div>";
      return;
    }

    const d3 = window.d3;
    const bounds = container.getBoundingClientRect();
    const width = Math.max(420, Math.floor(bounds.width || 0) - 4);
    const height = Math.max(360, Math.floor(bounds.height || 0) - 4);

    const data = buildHierarchy(options.root, options);
    const root = d3
      .hierarchy(data)
      .sum((d) => Number(d.value || 0))
      .sort((a, b) => b.value - a.value);

    d3
      .treemap()
      .size([width, height])
      .paddingOuter(3)
      .paddingTop(18)
      .paddingInner(2)
      .round(true)(root);

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height])
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("font", "12px Arial, sans-serif")
      .style("display", "block");

    const color = d3.scaleOrdinal(d3.quantize(d3.interpolateRainbow, root.children?.length + 1 || 8));

    const nodes = svg
      .selectAll("g")
      .data(root.descendants().filter((d) => d.depth > 0))
      .join("g")
      .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

    nodes
      .append("rect")
      .attr("rx", 4)
      .attr("ry", 4)
      .attr("width", (d) => Math.max(0, d.x1 - d.x0))
      .attr("height", (d) => Math.max(0, d.y1 - d.y0))
      .attr("fill", (d) => {
        let p = d;
        while (p.depth > 1) p = p.parent;
        return color(p.data.label);
      })
      .attr("fill-opacity", (d) => (d.children ? 0.72 : 0.58))
      .attr("stroke", (d) => (options.isSelected && options.isSelected(d.data.id) ? "#111827" : "#ffffff"))
      .attr("stroke-width", (d) => (options.isSelected && options.isSelected(d.data.id) ? 2 : 1))
      .style("cursor", "pointer")
      .on("click", (_, d) => {
        if (typeof options.onSelect === "function") {
          options.onSelect(d.data.id);
        }
      });

    nodes
      .append("title")
      .text((d) => `${d.data.id}\n${d.data.label}\nPapiers: ${Math.round(d.data.papers || 0)}`);

    nodes
      .append("text")
      .attr("x", 6)
      .attr("y", 14)
      .attr("fill", "#111827")
      .style("font-weight", 600)
      .style("pointer-events", "none")
      .text((d) => {
        const boxW = d.x1 - d.x0;
        const boxH = d.y1 - d.y0;
        if (boxW < 90 || boxH < 28) {
          return "";
        }
        return truncate(d.data.label, Math.max(10, Math.floor(boxW / 8)));
      });

    nodes
      .append("text")
      .attr("x", 6)
      .attr("y", 30)
      .attr("fill", "#1f2937")
      .style("font-size", "11px")
      .style("pointer-events", "none")
      .text((d) => {
        const boxW = d.x1 - d.x0;
        const boxH = d.y1 - d.y0;
        if (boxW < 120 || boxH < 45) {
          return "";
        }
        return `${d.data.id} · ${Math.round(d.data.papers || 0)} papiers`;
      });
  }

  window.TaxoTreemap = { render };
})();

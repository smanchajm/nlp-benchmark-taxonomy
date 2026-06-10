(function () {
  "use strict";

  function truncate(text, max = 28) {
    const raw = String(text || "");
    return raw.length <= max ? raw : `${raw.slice(0, max - 1)}…`;
  }

  function buildHierarchy(node, options) {
    const children = (node.children || []).map((c) => buildHierarchy(c, options));
    const directPapers = Math.max(0, Number(options.directPaperCount(node.id) || 0));
    if (directPapers > 0 && children.length > 0) {
      children.push({
        id: `${node.id}::__direct`,
        label: `${node.label} (direct)`,
        isDirect: true,
        value: directPapers,
        children: [],
      });
    }
    return {
      id: node.id,
      label: node.label,
      isDirect: false,
      value: children.length ? 0 : Math.max(0, Number(options.nodePaperCount(node.id) || 0)),
      children,
    };
  }

  function render(container, options) {
    container.innerHTML = "";
    if (typeof window.d3 === "undefined") {
      container.innerHTML = "<div>D3 introuvable pour sunburst.</div>";
      return;
    }
    const rootTaxo = options.root;
    if (!rootTaxo) {
      container.innerHTML = "<div>Aucune racine pour le sunburst.</div>";
      return;
    }
    const d3 = window.d3;
    const data = buildHierarchy(rootTaxo, options);
    const hierarchy = d3
      .hierarchy(data)
      .sum((d) => Number(d.value || 0))
      .sort((a, b) => b.value - a.value);

    const bounds = container.getBoundingClientRect();
    const width = Math.max(420, Math.floor(bounds.width || 0) - 4);
    const height = Math.max(360, Math.floor(bounds.height || 0) - 4);
    const radius = Math.max(90, Math.min(width, height) / 6 - 2);
    const partition = d3.partition().size([2 * Math.PI, hierarchy.height + 1]);
    const root = partition(hierarchy);
    root.each((d) => (d.current = d));

    const color = d3.scaleOrdinal(
      d3.quantize(d3.interpolateRainbow, root.children?.length + 1 || 10),
    );

    const svg = d3
      .select(container)
      .append("svg")
      .attr("viewBox", [-width / 2, -height / 2, width, height])
      .attr("width", width)
      .attr("height", height)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("font", "12px Arial, sans-serif");

    const g = svg.append("g");

    const arc = d3
      .arc()
      .startAngle((d) => d.x0)
      .endAngle((d) => d.x1)
      .padAngle((d) => Math.min((d.x1 - d.x0) / 2, 0.005))
      .padRadius(radius * 1.5)
      .innerRadius((d) => d.y0 * radius)
      .outerRadius((d) => Math.max(d.y0 * radius, d.y1 * radius - 1));

    const path = g
      .append("g")
      .selectAll("path")
      .data(root.descendants().slice(1))
      .join("path")
      .attr("fill", (d) => {
        if (d.data.isDirect) {
          return "#94a3b8";
        }
        let p = d;
        while (p.depth > 1) p = p.parent;
        return color(p.data.label);
      })
      .attr("fill-opacity", (d) => (arcVisible(d.current) ? (d.children ? 0.75 : 0.55) : 0))
      .attr("pointer-events", (d) => (arcVisible(d.current) ? "auto" : "none"))
      .attr("d", (d) => arc(d.current));

    path.filter((d) => d.children && !d.data.isDirect).style("cursor", "pointer").on("click", clicked);

    path.append("title").text(
      (d) =>
        `${d.data.id}\n${d.data.label}\nPapiers: ${Math.round(d.value || 0)}${
          d.data.isDirect ? "\n(assignations directes)" : ""
        }`,
    );

    const label = g
      .append("g")
      .attr("pointer-events", "none")
      .attr("text-anchor", "middle")
      .style("user-select", "none")
      .selectAll("text")
      .data(root.descendants().slice(1))
      .join("text")
      .attr("dy", "0.35em")
      .attr("fill-opacity", (d) => +labelVisible(d.current))
      .attr("transform", (d) => labelTransform(d.current))
      .text((d) =>
        d.data.isDirect
          ? truncate(d.data.label, 18)
          : truncate(`${d.data.label} (${Math.round(d.value || 0)})`, 24),
      );

    const parent = g
      .append("circle")
      .datum(root)
      .attr("r", radius)
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .style("cursor", "pointer")
      .on("click", clicked);
    parent.append("title").text("Zoom out");

    g
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .style("font-size", "13px")
      .style("fill", "#111827")
      .text("Zoom out");

    function arcVisible(d) {
      return d.y1 <= 3 && d.y0 >= 1 && d.x1 > d.x0;
    }

    function labelVisible(d) {
      return d.y1 <= 3 && d.y0 >= 1 && (d.x1 - d.x0) * (d.y1 - d.y0) > 0.04;
    }

    function labelTransform(d) {
      const x = ((d.x0 + d.x1) / 2) * (180 / Math.PI);
      const y = ((d.y0 + d.y1) / 2) * radius;
      return `rotate(${x - 90}) translate(${y},0) rotate(${x < 180 ? 0 : 180})`;
    }

    function clicked(_, p) {
      parent.datum(p.parent || root);
      root.each((d) => {
        d.target = {
          x0: Math.max(0, Math.min(1, (d.x0 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
          x1: Math.max(0, Math.min(1, (d.x1 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
          y0: Math.max(0, d.y0 - p.depth),
          y1: Math.max(0, d.y1 - p.depth),
        };
      });

      const t = g.transition().duration(650);
      path
        .transition(t)
        .tween("data", (d) => {
          const i = d3.interpolate(d.current, d.target);
          return (tt) => {
            d.current = i(tt);
          };
        })
        .filter(function (d) {
          return +this.getAttribute("fill-opacity") || arcVisible(d.target);
        })
        .attr("fill-opacity", (d) => (arcVisible(d.target) ? (d.children ? 0.75 : 0.55) : 0))
        .attr("pointer-events", (d) => (arcVisible(d.target) ? "auto" : "none"))
        .attrTween("d", (d) => () => arc(d.current));

      label
        .filter(function (d) {
          return +this.getAttribute("fill-opacity") || labelVisible(d.target);
        })
        .transition(t)
        .attr("fill-opacity", (d) => +labelVisible(d.target))
        .attrTween("transform", (d) => () => labelTransform(d.current));
    }
  }

  window.TaxoSunburst = { render };
})();

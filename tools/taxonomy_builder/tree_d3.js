(function () {
  "use strict";

  function truncate(text, max = 30) {
    const raw = String(text || "");
    return raw.length <= max ? raw : `${raw.slice(0, max - 1)}…`;
  }

  function buildVisibleTree(node, options) {
    if (!options.nodeOrDescendantMatches(node)) {
      return null;
    }
    const children = options
      .sortChildren(node.children || [])
      .map((child) => buildVisibleTree(child, options))
      .filter(Boolean);
    const leafChildren = options.showLeavesInTree
      ? options
          .assignedLeafNodesForTree(node.id)
          .filter((leaf) =>
            !options.query ? true : options.leafMatches(leaf, options.query),
          )
          .map((leaf) => ({
          id: `leaf::${leaf.leaf_id}`,
          leafId: leaf.leaf_id,
          label: leaf.label,
          description: leaf.description,
          isLeafAssignment: true,
          children: [],
        }))
      : [];
    return {
      id: node.id,
      label: node.label,
      description: node.description,
      isLeafAssignment: false,
      _collapsed: options.isCollapsed(node.id),
      children: options.isCollapsed(node.id) ? [] : [...children, ...leafChildren],
    };
  }

  function elbowPath(source, target) {
    const midY = (source.y + target.y) / 2;
    return `M${source.x},${source.y} V${midY} H${target.x} V${target.y}`;
  }

  function render(container, options) {
    container.innerHTML = "";
    if (typeof window.d3 === "undefined") {
      container.innerHTML =
        "<div>D3 introuvable (vendor/d3.v7.min.js manquant).</div>";
      return;
    }

    const treeData = buildVisibleTree(options.root, options);
    if (!treeData) {
      container.innerHTML = "<div>Aucun noeud ne correspond au filtre.</div>";
      return;
    }

    const d3 = window.d3;
    const root = d3.hierarchy(treeData);
    const layout = d3.tree().nodeSize([260, 170]);
    layout(root);

    let minX = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    root.each((n) => {
      minX = Math.min(minX, n.x);
      maxX = Math.max(maxX, n.x);
      maxY = Math.max(maxY, n.y);
    });

    const margin = { top: 36, right: 60, bottom: 40, left: 60 };
    const width = Math.max(1280, maxX - minX + margin.left + margin.right + 340);
    const height = Math.max(620, maxY + margin.top + margin.bottom + 180);
    const offsetX = margin.left - minX + 130;
    const offsetY = margin.top + 40;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("class", "d3-tree-svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`);

    const zoomLayer = svg.append("g");
    const graph = zoomLayer
      .append("g")
      .attr("transform", `translate(${offsetX}, ${offsetY})`);

    svg.call(
      d3
        .zoom()
        .scaleExtent([0.3, 2.4])
        .on("zoom", (event) => zoomLayer.attr("transform", event.transform)),
    );

    graph
      .selectAll("path.d3-link")
      .data(root.links())
      .enter()
      .append("path")
      .attr("class", "d3-link")
      .attr("d", (d) => elbowPath(d.source, d.target));

    const nodes = graph
      .selectAll("g.d3-node")
      .data(root.descendants())
      .enter()
      .append("g")
      .attr("class", (d) =>
        `d3-node${d.data.isLeafAssignment ? " leaf-assignment" : ""}${
          options.isSelected(d.data.id) ? " selected" : ""
        }`,
      )
      .attr("transform", (d) => `translate(${d.x}, ${d.y})`);

    nodes
      .append("rect")
      .attr("class", "d3-node-card")
      .attr("x", -118)
      .attr("y", -26)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("width", 236)
      .attr("height", 52)
      .on("click", (_, d) => {
        if (d.data.isLeafAssignment) {
          return;
        }
        options.onSelect(d.data.id);
      })
      .on("dblclick", (_, d) => {
        if (d.data.isLeafAssignment) {
          return;
        }
        options.onToggleCollapse(d.data.id);
      })
      .on("contextmenu", (event, d) => {
        if (d.data.isLeafAssignment) {
          return;
        }
        event.preventDefault();
        options.onContextMenu(event.clientX, event.clientY, d.data.id);
      })
      .on("dragover", function (event) {
        const datum = d3.select(this.parentNode).datum();
        if (datum?.data?.isLeafAssignment) {
          return;
        }
        event.preventDefault();
        d3.select(this.parentNode).classed("drop-target", true);
      })
      .on("dragleave", function () {
        d3.select(this.parentNode).classed("drop-target", false);
      })
      .on("drop", function (event, d) {
        if (d.data.isLeafAssignment) {
          return;
        }
        event.preventDefault();
        d3.select(this.parentNode).classed("drop-target", false);
        const leafId = event.dataTransfer.getData("application/x-taxo-leaf");
        const nodeId = event.dataTransfer.getData("application/x-taxo-node");
        if (leafId) {
          options.onLeafDrop(leafId, d.data.id);
          return;
        }
        if (nodeId) {
          options.onNodeDrop(nodeId, d.data.id);
        }
      });

    nodes
      .append("text")
      .attr("class", "d3-node-label")
      .attr("x", -110)
      .attr("y", -4)
      .text((d) =>
        d.data.isLeafAssignment
          ? truncate(`${d.data.leafId} · ${d.data.label}`, 38)
          : truncate(d.data.label, 38),
      );

    nodes
      .append("text")
      .attr("class", "d3-node-meta")
      .attr("x", -110)
      .attr("y", 14)
      .text((d) => {
        if (d.data.isLeafAssignment) {
          return "";
        }
        const direct = options.directAssignedCount(d.data.id);
        const sub = options.subtreeAssignedCount(d.data.id);
        return `direct:${direct} sous-arbre:${sub} enfants:${d.children?.length || 0}`;
      });

    nodes
      .append("circle")
      .attr("class", "d3-node-move-handle")
      .attr("cx", 108)
      .attr("cy", -18)
      .attr("r", 6)
      .attr("draggable", true)
      .on("dragstart", (event, d) => {
        if (d.data.isLeafAssignment) {
          event.dataTransfer.setData("application/x-taxo-leaf", d.data.leafId);
          return;
        }
        event.dataTransfer.setData("application/x-taxo-node", d.data.id);
      });

    nodes
      .append("title")
      .text((d) => `${d.data.id}\n${d.data.label}\n${d.data.description || "-"}`);
  }

  window.TaxoD3 = { render };
})();

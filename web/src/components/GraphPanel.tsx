import { useMemo, useEffect } from "react";
import { GitFork } from "lucide-react";
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  Handle,
  Position,
  useNodesState,
  useEdgesState
} from "@xyflow/react";
import type { Node, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "dagre";
import type { CausalScratchpad, CausalVariable, CausalEdge } from "../types";
import { useI18n } from "../i18n";
import "./GraphPanel.css";

interface GraphPanelProps {
  scratchpad: CausalScratchpad | null;
}

const ROLE_COLORS: Record<string, string> = {
  treatment: "#f7a24f",
  outcome: "#4ff7a2",
  confounder: "#f76f6f",
  mediator: "#b48cff",
  instrument: "#4fd6f7",
  other: "#3b82f6",
};

const EDGE_COLORS: Record<string, string> = {
  directed: "#3b82f6",
  undirected: "#8b94ab",
  bidirected: "#ef4444",
  partially_directed: "#eab308",
};

// Custom Node Component to mimic original "pill" design
interface CausalNodeData {
  name: string;
  role: string;
  color: string;
}

const CausalNode = ({ data }: { data: CausalNodeData }) => {
  return (
    <div 
      className="custom-causal-node"
      style={{ borderColor: data.color }}
    >
      <Handle type="target" position={Position.Left} />
      <span>{data.name}</span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
};

const nodeTypes = { causalNode: CausalNode };

const EMPTY_VARIABLES: CausalVariable[] = [];
const EMPTY_EDGES: CausalEdge[] = [];

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = "LR") => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  // Average size of the custom node
  const nodeWidth = 140;
  const nodeHeight = 50;

  dagreGraph.setGraph({ rankdir: direction, ranksep: 60, nodesep: 30 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const newNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    const newNode = { ...node };

    newNode.targetPosition = Position.Left;
    newNode.sourcePosition = Position.Right;

    // Shift anchor from center to top-left for React Flow
    newNode.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    return newNode;
  });

  return { nodes: newNodes, edges };
};

export function GraphPanel({ scratchpad }: GraphPanelProps) {
  const { t } = useI18n();
  const variables = scratchpad?.variables ?? EMPTY_VARIABLES;
  const rawEdges = scratchpad?.edges ?? EMPTY_EDGES;
  const hasVars = variables.length > 0;
  const hasEdges = rawEdges.length > 0;

  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    const nodeNames = variables.map((v) => v.name);
    // If no variables defined but edges exist, collect node names from edges
    if (nodeNames.length === 0 && rawEdges.length > 0) {
      const nodeSet = new Set<string>();
      rawEdges.forEach((e) => {
        if (e.from) nodeSet.add(e.from);
        if (e.to) nodeSet.add(e.to);
      });
      nodeNames.push(...Array.from(nodeSet));
    }

    const flowNodes: Node[] = nodeNames.map((name) => {
      const variable = variables.find((v) => v.name === name);
      const role = variable?.role ?? "other";
      return {
        id: name,
        type: "causalNode",
        data: { name, role, color: ROLE_COLORS[role] ?? ROLE_COLORS.other },
        position: { x: 0, y: 0 },
      };
    });

    const flowEdges: Edge[] = rawEdges.map((e, idx) => {
      const isDirected = e.edgeType === "directed";
      const isUndirected = e.edgeType === "undirected";
      const isBidirected = e.edgeType === "bidirected";
      const isPartial = e.edgeType === "partially_directed";

      const color = EDGE_COLORS[e.edgeType] ?? EDGE_COLORS.directed;

      return {
        id: `e-${e.from}-${e.to}-${idx}`,
        source: e.from,
        target: e.to,
        type: "default",
        animated: false,
        style: {
          stroke: color,
          strokeWidth: 1.75,
          strokeDasharray: isUndirected ? "5 5" : undefined,
        },
        markerEnd: isDirected || isBidirected || isPartial ? {
          type: MarkerType.ArrowClosed,
          color: color,
        } : undefined,
        markerStart: isBidirected ? {
          type: MarkerType.ArrowClosed,
          color: color,
        } : undefined,
      };
    });

    return getLayoutedElements(flowNodes, flowEdges, "LR");
  }, [variables, rawEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Sync state when props/layout changes
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  return (
    <div className="graph-panel">
      <div className="graph-header">
        <div className="flex items-center gap-2">
          <GitFork size={14} style={{ color: "var(--accent)" }} />
          <span className="font-semibold" style={{ fontSize: 13 }}>{t("graph.title")}</span>
        </div>
        <div className="flex items-center gap-2">
          {hasVars && <span className="badge badge-accent">{t("graph.vars", { count: variables.length })}</span>}
          {hasEdges && <span className="badge badge-muted">{t("graph.edges", { count: rawEdges.length })}</span>}
          {scratchpad?.score !== undefined && (
            <span className="badge badge-green">{t("graph.score", { score: scratchpad.score.toFixed(1) })}</span>
          )}
        </div>
      </div>

      <div className="graph-canvas-container">
        {hasVars || hasEdges ? (
          <div className="graph-canvas-wrapper">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodeTypes={nodeTypes}
              fitView
              attributionPosition="bottom-right"
              minZoom={0.1}
            >
              <Background gap={12} size={1} />
              <Controls showInteractive={false} />
            </ReactFlow>
          </div>
        ) : (
          <div className="graph-empty">
            <GitFork size={28} style={{ opacity: 0.25 }} />
            <div>{t("graph.empty")}</div>
            <div className="text-sm text-muted">{t("graph.emptySub")}</div>
          </div>
        )}
      </div>

      {/* Legend */}
      {(hasVars || hasEdges) && (
        <div className="graph-legend">
          <span className="legend-item"><span className="legend-line directed" />{t("graph.directed")}</span>
          <span className="legend-item"><span className="legend-line undirected" />{t("graph.undirected")}</span>
          <span className="legend-item"><span className="legend-line bidirected" />{t("graph.bidirected")}</span>
          <span className="legend-item"><span className="legend-line partial" />{t("graph.partial")}</span>
        </div>
      )}
    </div>
  );
}

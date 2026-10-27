import { useCallback, useEffect, useMemo, useState } from "react";
import {
	Background,
	BaseEdge,
	Controls,
	EdgeLabelRenderer,
	Handle,
	MarkerType,
	Position,
	ReactFlow,
	useEdgesState,
	useNodesState,
} from "@xyflow/react";
import type { Edge, EdgeProps, Node, NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "dagre";
import { BrainCircuit, RefreshCw, Search } from "lucide-react";
import { api } from "../api";
import { useI18n } from "../i18n";
import type {
	CausalCognitionIndexConcept,
	CausalCognitionIndexRelation,
	CausalCognitionIndexResponse,
	CausalCognitionRelationClaim,
} from "../types";
import "./CognitionIndexPanel.css";

type CognitionConceptNodeData = {
	concept: CausalCognitionIndexConcept;
	matched: boolean;
};

type CognitionRelationEdgeData = {
	relation: CausalCognitionIndexRelation;
	label: string;
	offset: number;
};

type CognitionConceptNode = Node<CognitionConceptNodeData, "cognitionConcept">;
type CognitionRelationEdge = Edge<
	CognitionRelationEdgeData,
	"cognitionRelation"
>;

type CognitionSelection =
	| { kind: "concept"; id: string }
	| { kind: "relation"; id: string };

const NODE_WIDTH = 172;
const NODE_HEIGHT = 58;

function CognitionNode({ data, selected }: NodeProps<CognitionConceptNode>) {
	const { concept, matched } = data;
	return (
		<div
			className={`cognition-graph-node ${matched ? "matched" : ""} ${selected ? "selected" : ""}`}
		>
			<Handle type="target" position={Position.Left} />
			<div className="cognition-graph-node-label">{concept.label}</div>
			<div className="cognition-graph-node-degree">
				<span>← {concept.activeIncoming}</span>
				<span>→ {concept.activeOutgoing}</span>
			</div>
			<Handle type="source" position={Position.Right} />
		</div>
	);
}

function CognitionEdge({
	id,
	sourceX,
	sourceY,
	targetX,
	targetY,
	markerEnd,
	style,
	data,
	selected,
}: EdgeProps<CognitionRelationEdge>) {
	const dx = targetX - sourceX;
	const dy = targetY - sourceY;
	const length = Math.max(Math.hypot(dx, dy), 1);
	const offset = data?.offset ?? 0;
	const controlX = (sourceX + targetX) / 2 + (-dy / length) * offset;
	const controlY = (sourceY + targetY) / 2 + (dx / length) * offset;
	const labelX = 0.25 * sourceX + 0.5 * controlX + 0.25 * targetX;
	const labelY = 0.25 * sourceY + 0.5 * controlY + 0.25 * targetY;
	const path = `M ${sourceX},${sourceY} Q ${controlX},${controlY} ${targetX},${targetY}`;

	return (
		<>
			<BaseEdge
				id={id}
				path={path}
				markerEnd={markerEnd}
				style={{ ...style, strokeWidth: selected ? 3 : style?.strokeWidth }}
				interactionWidth={24}
			/>
			<EdgeLabelRenderer>
				<div
					className={`cognition-edge-label ${selected ? "selected" : ""}`}
					style={{
						transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
					}}
				>
					{data?.label}
				</div>
			</EdgeLabelRenderer>
		</>
	);
}

const nodeTypes = { cognitionConcept: CognitionNode };
const edgeTypes = { cognitionRelation: CognitionEdge };

function relationColor(relation: CausalCognitionIndexRelation): string {
	if (!relation.assessment.active) return "#7d879d";
	return "#4f86f7";
}

function relationLabel(relation: CausalCognitionIndexRelation): string {
	const posterior = Math.round(relation.assessment.posterior * 100);
	const lag =
		relation.lag && relation.lag !== "unknown" ? ` · ${relation.lag}` : "";
	return `${posterior}%${lag}`;
}

function layoutGraph(
	nodes: CognitionConceptNode[],
	edges: CognitionRelationEdge[],
): { nodes: CognitionConceptNode[]; edges: CognitionRelationEdge[] } {
	const graph = new dagre.graphlib.Graph({ multigraph: true });
	graph.setDefaultEdgeLabel(() => ({}));
	graph.setGraph({
		rankdir: "LR",
		ranksep: 105,
		nodesep: 48,
		edgesep: 30,
		marginx: 26,
		marginy: 26,
	});
	for (const node of nodes) {
		graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
	}
	for (const edge of edges) {
		graph.setEdge(edge.source, edge.target, {}, edge.id);
	}
	dagre.layout(graph);

	return {
		nodes: nodes.map((node) => {
			const position = graph.node(node.id);
			return {
				...node,
				position: {
					x: position.x - NODE_WIDTH / 2,
					y: position.y - NODE_HEIGHT / 2,
				},
				sourcePosition: Position.Right,
				targetPosition: Position.Left,
			};
		}),
		edges,
	};
}

function searchText(value: string): string {
	return value
		.toLocaleLowerCase()
		.replace(/[_-]+/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function relationSearchText(relation: CausalCognitionIndexRelation): string {
	return searchText(
		[
			relation.id,
			relation.mechanism,
			relation.lag,
			...relation.contexts,
			...relation.claims.flatMap((claim) => [
				claim.verdict,
				claim.confidence,
				claim.note,
			]),
		]
			.filter(Boolean)
			.join(" "),
	);
}

function graphElements(
	data: CausalCognitionIndexResponse | null,
	query: string,
): {
	nodes: CognitionConceptNode[];
	edges: CognitionRelationEdge[];
	visibleConceptIds: Set<string>;
	visibleRelationIds: Set<string>;
} {
	if (!data) {
		return {
			nodes: [],
			edges: [],
			visibleConceptIds: new Set(),
			visibleRelationIds: new Set(),
		};
	}
	const normalized = searchText(query.trim());
	const matchedConceptIds = new Set<string>();
	const matchedRelationIds = new Set<string>();
	if (normalized) {
		for (const concept of data.concepts) {
			const haystack = searchText(
				[
					concept.id,
					concept.label,
					concept.description,
					...concept.aliases,
					...concept.keywords,
				]
					.filter(Boolean)
					.join(" "),
			);
			if (haystack.includes(normalized)) matchedConceptIds.add(concept.id);
		}
		for (const relation of data.relations) {
			if (relationSearchText(relation).includes(normalized)) {
				matchedRelationIds.add(relation.id);
			}
		}
	}

	const visibleConceptIds = new Set<string>();
	const visibleRelationIds = new Set<string>();
	if (!normalized) {
		for (const concept of data.concepts) visibleConceptIds.add(concept.id);
		for (const relation of data.relations) visibleRelationIds.add(relation.id);
	} else {
		for (const conceptId of matchedConceptIds) visibleConceptIds.add(conceptId);
		for (const relation of data.relations) {
			if (
				matchedRelationIds.has(relation.id) ||
				matchedConceptIds.has(relation.causeId) ||
				matchedConceptIds.has(relation.effectId)
			) {
				visibleRelationIds.add(relation.id);
				visibleConceptIds.add(relation.causeId);
				visibleConceptIds.add(relation.effectId);
			}
		}
	}

	const visibleRelations = data.relations.filter((relation) =>
		visibleRelationIds.has(relation.id),
	);
	const pairGroups = new Map<string, CausalCognitionIndexRelation[]>();
	for (const relation of visibleRelations) {
		const pairKey = [relation.causeId, relation.effectId].sort().join("::");
		const group = pairGroups.get(pairKey) ?? [];
		group.push(relation);
		pairGroups.set(pairKey, group);
	}
	const offsets = new Map<string, number>();
	for (const relations of pairGroups.values()) {
		relations.sort((left, right) => left.id.localeCompare(right.id));
		relations.forEach((relation, index) => {
			const baseOffset = (index - (relations.length - 1) / 2) * 34;
			const canonicalFirst = [relation.causeId, relation.effectId].sort()[0];
			const direction = relation.causeId === canonicalFirst ? 1 : -1;
			offsets.set(relation.id, baseOffset * direction);
		});
	}

	const nodes: CognitionConceptNode[] = data.concepts
		.filter((concept) => visibleConceptIds.has(concept.id))
		.map((concept) => ({
			id: concept.id,
			type: "cognitionConcept",
			position: { x: 0, y: 0 },
			data: { concept, matched: matchedConceptIds.has(concept.id) },
		}));
	const edges: CognitionRelationEdge[] = visibleRelations.map((relation) => {
		const color = relationColor(relation);
		return {
			id: relation.id,
			source: relation.causeId,
			target: relation.effectId,
			type: "cognitionRelation",
			data: {
				relation,
				label: relationLabel(relation),
				offset: offsets.get(relation.id) ?? 0,
			},
			style: {
				stroke: color,
				strokeWidth: 1.6 + relation.assessment.posterior,
				strokeDasharray: relation.assessment.active ? undefined : "6 5",
				opacity: relation.assessment.active ? 0.9 : 0.52,
			},
			markerEnd: {
				type: MarkerType.ArrowClosed,
				color,
				width: 16,
				height: 16,
			},
		};
	});
	return {
		...layoutGraph(nodes, edges),
		visibleConceptIds,
		visibleRelationIds,
	};
}

export function CognitionIndexPanel() {
	const { locale } = useI18n();
	const [data, setData] = useState<CausalCognitionIndexResponse | null>(null);
	const [query, setQuery] = useState("");
	const [selection, setSelection] = useState<CognitionSelection | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const load = useCallback(async () => {
		setLoading(true);
		setError(null);
		try {
			setData(await api.getCausalCognitionIndex());
		} catch (reason) {
			setError(reason instanceof Error ? reason.message : String(reason));
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		void load();
	}, [load]);

	const graph = useMemo(() => graphElements(data, query), [data, query]);
	const [nodes, setNodes, onNodesChange] = useNodesState<CognitionConceptNode>(
		graph.nodes,
	);
	const [edges, setEdges, onEdgesChange] = useEdgesState<CognitionRelationEdge>(
		graph.edges,
	);

	useEffect(() => {
		setNodes(graph.nodes);
		setEdges(graph.edges);
	}, [graph, setEdges, setNodes]);

	useEffect(() => {
		if (
			selection?.kind === "relation" &&
			graph.visibleRelationIds.has(selection.id)
		) {
			return;
		}
		if (
			selection?.kind === "concept" &&
			graph.visibleConceptIds.has(selection.id)
		) {
			return;
		}
		const relation = graph.edges[0];
		const concept = graph.nodes[0];
		setSelection(
			relation
				? { kind: "relation", id: relation.id }
				: concept
					? { kind: "concept", id: concept.id }
					: null,
		);
	}, [graph, selection]);

	const selectedRelation =
		selection?.kind === "relation"
			? data?.relations.find((relation) => relation.id === selection.id)
			: undefined;
	const selectedConcept =
		selection?.kind === "concept"
			? data?.concepts.find((concept) => concept.id === selection.id)
			: undefined;
	const conceptLabels = useMemo(
		() => new Map(data?.concepts.map((concept) => [concept.id, concept.label])),
		[data],
	);

	const text =
		locale === "zh"
			? {
					title: "Causal Cognition Graph",
					subtitle: "上下文限定的持久化因果知识网络",
					refresh: "刷新",
					search: "搜索概念、别名、关键词、机制或 context…",
					concepts: "概念",
					variants: "活跃 / 全部关系",
					claims: "Relation Claims",
					episodes: "Episodes",
					empty: "这个来源目前还没有 Causal Cognition 图。",
					noMatch: "没有匹配当前搜索的概念或关系。",
					loading: "正在读取 Causal Cognition…",
					graphHint: "点击节点或有向边查看完整语义",
					inspector: "图元素详情",
					mechanism: "机制",
					contexts: "适用语境",
					lag: "时间滞后",
					posterior: "Posterior",
					logOdds: "Posterior log-odds",
					prior: "Prior log-odds",
					support: "Support log BF",
					contradiction: "Contradiction log BF",
					activeRelation: "可检索",
					inactiveRelation: "低于检索阈值",
					causalRelation: "有向因果关系",
					aliases: "别名",
					keywords: "Index 关键词",
					description: "概念说明",
					degree: "活跃入度 / 出度",
					claimsTitle: "独立 Claims",
					noClaims: "暂无 claim",
					selectHint: "选择一个节点或关系查看详情。",
				}
			: {
					title: "Causal Cognition Graph",
					subtitle: "A context-qualified persistent causal knowledge network",
					refresh: "Refresh",
					search:
						"Search concepts, aliases, keywords, mechanisms, or contexts…",
					concepts: "Concepts",
					variants: "Active / all relations",
					claims: "Relation claims",
					episodes: "Episodes",
					empty: "This source does not contain a Causal Cognition graph yet.",
					noMatch: "No concept or relation matches the current search.",
					loading: "Loading Causal Cognition…",
					graphHint: "Select a node or directed edge to inspect its semantics",
					inspector: "Graph element",
					mechanism: "Mechanism",
					contexts: "Contexts",
					lag: "Lag",
					posterior: "Posterior",
					logOdds: "Posterior log-odds",
					prior: "Prior log-odds",
					support: "Support log BF",
					contradiction: "Contradiction log BF",
					activeRelation: "Retrievable",
					inactiveRelation: "Below retrieval threshold",
					causalRelation: "Directed causal relation",
					aliases: "Aliases",
					keywords: "Index keywords",
					description: "Description",
					degree: "Active in / out degree",
					claimsTitle: "Independent claims",
					noClaims: "No claims",
					selectHint: "Select a node or relation to inspect it.",
				};

	return (
		<div className="cognition-index-panel">
			<div className="cognition-index-header">
				<div className="cognition-index-heading">
					<BrainCircuit size={17} />
					<div>
						<div className="cognition-index-title">{text.title}</div>
						<div className="cognition-index-subtitle">{text.subtitle}</div>
					</div>
				</div>
				<button
					className="btn btn-ghost btn-icon cognition-refresh"
					onClick={() => void load()}
					title={text.refresh}
					disabled={loading}
				>
					<RefreshCw size={14} className={loading ? "cognition-spin" : ""} />
				</button>
			</div>

			{data && (
				<div className="cognition-stats">
					<Stat label={text.concepts} value={data.counts.concepts} />
					<Stat
						label={text.variants}
						value={`${data.counts.activeRelations}/${data.counts.relationVariants}`}
					/>
					<Stat label={text.claims} value={data.counts.relationClaims} />
					<Stat label={text.episodes} value={data.counts.episodes} />
				</div>
			)}

			<div className="cognition-search-wrap">
				<Search size={14} />
				<input
					value={query}
					onChange={(event) => setQuery(event.target.value)}
					placeholder={text.search}
					aria-label={text.search}
				/>
				{data && (
					<span>
						{graph.nodes.length}/{data.counts.concepts}
					</span>
				)}
			</div>

			<div className="cognition-index-body">
				{loading ? (
					<div className="cognition-empty">
						<span className="spinner" />
						{text.loading}
					</div>
				) : error ? (
					<div className="cognition-empty cognition-error">{error}</div>
				) : !data || data.counts.concepts === 0 ? (
					<div className="cognition-empty">{text.empty}</div>
				) : graph.nodes.length === 0 ? (
					<div className="cognition-empty">{text.noMatch}</div>
				) : (
					<>
						<div className="cognition-graph-shell">
							<div className="cognition-graph-hint">{text.graphHint}</div>
							<ReactFlow
								key={`${data.source}:${data.updatedAt ?? "none"}:${query}`}
								nodes={nodes}
								edges={edges}
								onNodesChange={onNodesChange}
								onEdgesChange={onEdgesChange}
								onNodeClick={(_, node) =>
									setSelection({ kind: "concept", id: node.id })
								}
								onEdgeClick={(_, edge) =>
									setSelection({ kind: "relation", id: edge.id })
								}
								nodeTypes={nodeTypes}
								edgeTypes={edgeTypes}
								fitView
								fitViewOptions={{ padding: 0.18, maxZoom: 1.15 }}
								minZoom={0.08}
								maxZoom={1.8}
								nodesConnectable={false}
								elementsSelectable
								proOptions={{ hideAttribution: true }}
							>
								<Background gap={16} size={1} />
								<Controls showInteractive={false} />
							</ReactFlow>
						</div>
						<div className="cognition-inspector">
							<div className="cognition-inspector-title">{text.inspector}</div>
							{selectedRelation ? (
								<RelationInspector
									relation={selectedRelation}
									causeLabel={
										conceptLabels.get(selectedRelation.causeId) ??
										selectedRelation.causeId
									}
									effectLabel={
										conceptLabels.get(selectedRelation.effectId) ??
										selectedRelation.effectId
									}
									text={text}
									locale={locale}
								/>
							) : selectedConcept ? (
								<ConceptInspector concept={selectedConcept} text={text} />
							) : (
								<div className="cognition-inspector-empty">
									{text.selectHint}
								</div>
							)}
						</div>
					</>
				)}
			</div>

			{data && (
				<div className="cognition-index-footer">
					<span>{data.sourceLabel}</span>
					<code>{data.storageRelativePath}</code>
					<span>schema v{data.schemaVersion}</span>
				</div>
			)}
		</div>
	);
}

type InspectorText = ReturnType<typeof inspectorTextShape>;

function inspectorTextShape() {
	return {
		mechanism: "",
		contexts: "",
		lag: "",
		posterior: "",
		logOdds: "",
		prior: "",
		support: "",
		contradiction: "",
		activeRelation: "",
		inactiveRelation: "",
		causalRelation: "",
		aliases: "",
		keywords: "",
		description: "",
		degree: "",
		claimsTitle: "",
		noClaims: "",
	};
}

function RelationInspector({
	relation,
	causeLabel,
	effectLabel,
	text,
	locale,
}: {
	relation: CausalCognitionIndexRelation;
	causeLabel: string;
	effectLabel: string;
	text: InspectorText;
	locale: string;
}) {
	const assessment = relation.assessment;
	return (
		<div className="cognition-inspector-content">
			<div className="cognition-relation-heading">
				<strong>{causeLabel}</strong>
				<span>→</span>
				<strong>{effectLabel}</strong>
			</div>
			<div className="cognition-relation-meta">
				<span className="causal">{text.causalRelation}</span>
				<span className={assessment.active ? "active" : "inactive"}>
					{assessment.active ? text.activeRelation : text.inactiveRelation}
				</span>
			</div>

			<div className="cognition-posterior">
				<div>
					<span>{text.posterior}</span>
					<strong>{(assessment.posterior * 100).toFixed(1)}%</strong>
				</div>
				<div className="cognition-posterior-track">
					<span style={{ width: `${assessment.posterior * 100}%` }} />
					<i />
				</div>
			</div>

			<div className="cognition-assessment-grid">
				<Metric label={text.logOdds} value={assessment.logOdds} signed />
				<Metric label={text.prior} value={assessment.priorLogOdds} signed />
				<Metric label={text.support} value={assessment.supportLogBayesFactor} />
				<Metric
					label={text.contradiction}
					value={assessment.contradictionLogBayesFactor}
				/>
			</div>

			<InspectorField label={text.mechanism} value={relation.mechanism} />
			<InspectorField label={text.contexts} values={relation.contexts} />
			<InspectorField label={text.lag} value={relation.lag} />

			<div className="cognition-claims-title">
				{text.claimsTitle} <span>{relation.claims.length}</span>
			</div>
			{relation.claims.length === 0 ? (
				<div className="cognition-inspector-empty">{text.noClaims}</div>
			) : (
				<div className="cognition-claim-list">
					{relation.claims.map((claim) => (
						<ClaimCard
							key={`${relation.id}:${claim.episodeId}`}
							claim={claim}
							locale={locale}
						/>
					))}
				</div>
			)}
			<code className="cognition-element-id">{relation.id}</code>
		</div>
	);
}

function ConceptInspector({
	concept,
	text,
}: {
	concept: CausalCognitionIndexConcept;
	text: InspectorText;
}) {
	return (
		<div className="cognition-inspector-content">
			<div className="cognition-concept-heading">{concept.label}</div>
			<InspectorField label={text.description} value={concept.description} />
			<InspectorField label={text.aliases} values={concept.aliases} />
			<InspectorField label={text.keywords} values={concept.keywords} />
			<InspectorField
				label={text.degree}
				value={`← ${concept.activeIncoming}   → ${concept.activeOutgoing}`}
			/>
			<code className="cognition-element-id">{concept.id}</code>
		</div>
	);
}

function ClaimCard({
	claim,
	locale,
}: {
	claim: CausalCognitionRelationClaim;
	locale: string;
}) {
	const verdict =
		locale === "zh"
			? {
					hypothesized: "假设",
					supported: "支持",
					contradicted: "反驳",
				}[claim.verdict]
			: claim.verdict;
	return (
		<div className={`cognition-claim ${claim.verdict}`}>
			<div className="cognition-claim-topline">
				<span>{verdict}</span>
				<strong>{claim.confidence ?? "high"}</strong>
			</div>
			{claim.note && <p>{claim.note}</p>}
			<code>{claim.episodeId}</code>
		</div>
	);
}

function InspectorField({
	label,
	value,
	values,
}: {
	label: string;
	value?: string;
	values?: string[];
}) {
	if (!value && (!values || values.length === 0)) return null;
	return (
		<div className="cognition-inspector-field">
			<label>{label}</label>
			{values ? (
				<div className="cognition-inspector-tags">
					{values.map((item) => (
						<span key={item}>{item}</span>
					))}
				</div>
			) : (
				<p>{value}</p>
			)}
		</div>
	);
}

function Metric({
	label,
	value,
	signed = false,
}: {
	label: string;
	value: number;
	signed?: boolean;
}) {
	return (
		<div className="cognition-assessment-metric">
			<span>{label}</span>
			<strong>
				{signed && value > 0 ? "+" : ""}
				{value.toFixed(2)}
			</strong>
		</div>
	);
}

function Stat({ label, value }: { label: string; value: number | string }) {
	return (
		<div className="cognition-stat">
			<strong>{value}</strong>
			<span>{label}</span>
		</div>
	);
}

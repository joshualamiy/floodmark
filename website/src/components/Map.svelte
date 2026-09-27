<script lang="ts">
	import { onMount } from "svelte";
	import { createQuery } from "@tanstack/svelte-query";
	import {
		Map as MapLibreMap,
		setWorkerUrl,
		GeoJSONSource,
		type StyleSpecification,
	} from "maplibre-gl";
 	// ?worker&url bundles the worker with its "./maplibre-gl-shared.mjs" import into one file;
	// a plain ?url copied only the worker, so it failed to load once deployed.
	import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
	import "maplibre-gl/dist/maplibre-gl.css";
	import { api } from "$lib/api";
	import { useMapState } from "$lib/state/map.svelte";
	import type { Camera } from "$lib/types/camera";
	import Button from "./ui/button/button.svelte";
	import MapFilter, {
		type MapAvailabilityFilter,
		type MapFilterOption as MapFilterValue,
		type MapFreshnessFilter,
	} from "./map-filter.svelte";
	import Plus from "@lucide/svelte/icons/plus";
	import Minus from "@lucide/svelte/icons/minus";

	setWorkerUrl(maplibreWorkerUrl);

	let { onCameraSelect }: { onCameraSelect?: () => void } = $props();

	const mapState = useMapState();
	let map: MapLibreMap | undefined;
	let filter = $state<MapFilterValue>("all");
	let freshness = $state<MapFreshnessFilter>("all");
	let availability = $state<MapAvailabilityFilter>("all");
	const camerasQuery = createQuery(() => ({
		queryKey: ["cameras"],
		queryFn: () => api().cameras.list(),
	}));
	let cameras = $derived(camerasQuery.data ?? []);
	const cameraSourceId = "cameras";
	const cameraLayerId = "camera-points";
	const clusterLayerId = "camera-clusters";
	const clusterCountLayerId = "camera-cluster-count";
	type CameraFeatureCollection = {
		type: "FeatureCollection";
		features: Array<{
			type: "Feature";
			geometry: { type: "Point"; coordinates: [number, number] };
			properties: {
				id: string;
				name: string;
				location: string;
				alertStatus: string | null;
			};
		}>;
	};

	function cameraData(): CameraFeatureCollection {
		return {
			type: "FeatureCollection",
			features: cameras.flatMap((camera) => {
				if (!matchesFilters(camera)) return [];

				const latitude = Number(camera.latitude);
				const longitude = Number(camera.longitude);
				if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return [];

				return [
					{
						type: "Feature",
						geometry: { type: "Point", coordinates: [longitude, latitude] },
						properties: {
							id: camera.id,
							name: camera.name,
							alertStatus: camera.latestImage?.alertStatus ?? null,
							location: [camera.locationDescription, camera.roadway, camera.direction]
								.filter(Boolean)
								.join(" | "),
						},
					},
				];
			}),
		};
	}

	function matchesFilter(alertStatus: string | null): boolean {
		if (filter === "all") return true;
		if (filter === "flooded") return alertStatus === "flooded";
		if (filter === "wet") return alertStatus === "wet";
		if (filter === "clear") return alertStatus === "dry";
		return alertStatus === null;
	}

	function matchesFilters(camera: Camera): boolean {
		if (!matchesFilter(camera.latestImage?.alertStatus ?? null)) return false;

		const processingStatus = camera.latestImage?.processingStatus;
		const hasProcessingError = processingStatus === "error";
		const isUnavailable =
			camera.latestImage === null ||
			processingStatus === "skipped" ||
			processingStatus === "unprocessed";

		const capturedAt = camera.latestImage?.capturedAt;
		const minutesSinceCapture = capturedAt
			? (Date.now() - new Date(capturedAt).getTime()) / 60000
			: null;
		const hasUsableImage = camera.latestImage !== null && !isUnavailable && !hasProcessingError;
		if (freshness !== "all" && !hasUsableImage) return false;
		if (freshness === "recent" && (minutesSinceCapture === null || minutesSinceCapture >= 20))
			return false;
		if (freshness === "stale" && (minutesSinceCapture === null || minutesSinceCapture < 20))
			return false;
		if (freshness === "no-capture" && minutesSinceCapture !== null) return false;

		if (availability === "error" && !hasProcessingError) return false;
		if (availability === "unavailable" && !isUnavailable) return false;
		if (availability === "available" && (isUnavailable || hasProcessingError)) return false;

		return true;
	}

	function updateCameraSource() {
		const currentMap = map;
		if (!currentMap) return;

		const source = currentMap.getSource(cameraSourceId);
		if (source instanceof GeoJSONSource) void source.setData(cameraData());
	}

	function zoomIn() {
		map?.zoomIn({ duration: 200 });
	}

	function zoomOut() {
		map?.zoomOut({ duration: 200 });
	}

	function focusCamera(currentMap: MapLibreMap, camera: Camera) {
		const latitude = Number(camera.latitude);
		const longitude = Number(camera.longitude);
		if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;

		currentMap.easeTo({
			center: [longitude, latitude],
			zoom: Math.max(currentMap.getZoom(), 13),
			duration: 400,
		});
	}

	$effect(() => {
		if (!mapState.loading) updateCameraSource();
	});

	$effect(() => {
		const currentMap = map;
		const camera = mapState.activeCamera;
		if (!currentMap || !camera || mapState.loading) return;

		focusCamera(currentMap, camera);
	});

	onMount(() => {
		let cancelled = false;

		void fetch(
			"https://raw.githubusercontent.com/openmaptiles/osm-bright-gl-style/master/style.json",
		)
			.then((response) => {
				if (!response.ok) throw new Error(`Unable to load map style: ${response.status}`);
				return response.json();
			})
			.then((style: StyleSpecification) => {
				if (cancelled) return;

				const source = style.sources.openmaptiles;
				if (!source || source.type !== "vector" || !("url" in source)) {
					throw new Error("OSM Bright style has no compatible OpenMapTiles source");
				}

				source.url = "https://tiles.openfreemap.org/planet";
				style.glyphs = "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf";
				for (const layer of style.layers) {
					const layerId = layer.id;
					const isRoadLayer =
						layerId.startsWith("highway-") ||
						layerId.startsWith("bridge-") ||
						layerId.startsWith("tunnel-") ||
						layerId.startsWith("road_");
					if (!isRoadLayer || layer.type !== "line" || !("paint" in layer) || !layer.paint)
						continue;

					const paint = layer.paint as { "line-color"?: unknown };
					if (!("line-color" in paint)) continue;
					paint["line-color"] = layerId.includes("casing") ? "#6f8797" : "#c3d2db";
				}

				const currentMap = new MapLibreMap({
					container: mapState.container!,
					style,
					maxBounds: [
						[-85.2, 33.1],
						[-83.35, 34.5],
					],
					renderWorldCopies: false,
					attributionControl: {
						customAttribution: '<a href="https://511ga.org//">511ga.org</a>',
					},
					center: mapState.position,
					zoom: mapState.zoom,
				});
				map = currentMap;

				currentMap.on("load", () => {
					currentMap.addSource(cameraSourceId, {
						type: "geojson",
						data: cameraData(),
						cluster: true,
						clusterMaxZoom: 14,
						clusterRadius: 50,
						clusterProperties: {
							floodedPriority: ["+", ["case", ["==", ["get", "alertStatus"], "flooded"], 100, 0]],
							wetPriority: ["+", ["case", ["==", ["get", "alertStatus"], "wet"], 10, 0]],
							clearPriority: ["+", ["case", ["==", ["get", "alertStatus"], "dry"], 1, 0]],
						},
					});
					currentMap.addLayer({
						id: clusterLayerId,
						type: "circle",
						source: cameraSourceId,
						filter: ["has", "point_count"],
						paint: {
							"circle-color": [
								"case",
								[
									"all",
									[">", ["coalesce", ["get", "floodedPriority"], 0], 0],
									[
										">=",
										["coalesce", ["get", "floodedPriority"], 0],
										["coalesce", ["get", "wetPriority"], 0],
									],
									[
										">=",
										["coalesce", ["get", "floodedPriority"], 0],
										["coalesce", ["get", "clearPriority"], 0],
									],
								],
								"#dc2626",
								[
									"all",
									[">", ["coalesce", ["get", "wetPriority"], 0], 0],
									[
										">=",
										["coalesce", ["get", "wetPriority"], 0],
										["coalesce", ["get", "clearPriority"], 0],
									],
								],
								"#facc15",
								[">", ["coalesce", ["get", "clearPriority"], 0], 0],
								"#00A6AD",
								"#94a3b8",
							],
							"circle-radius": ["step", ["get", "point_count"], 16, 100, 20, 750, 24],
							"circle-stroke-color": "#ffffff",
							"circle-stroke-width": 2,
						},
					});
					currentMap.addLayer({
						id: clusterCountLayerId,
						type: "symbol",
						source: cameraSourceId,
						filter: ["has", "point_count"],
						layout: {
							"text-field": ["get", "point_count_abbreviated"],
							"text-size": 12,
						},
						paint: { "text-color": "#ffffff" },
					});
					currentMap.addLayer({
						id: cameraLayerId,
						type: "circle",
						source: cameraSourceId,
						filter: ["!", ["has", "point_count"]],
						paint: {
							"circle-color": [
								"case",
								["==", ["get", "alertStatus"], "flooded"],
								"#dc2626",
								["==", ["get", "alertStatus"], "wet"],
								"#facc15",
								["==", ["get", "alertStatus"], "dry"],
								"#00A6AD",
								"#94a3b8",
							],
							"circle-radius": 10,
							"circle-stroke-color": "#ffffff",
							"circle-stroke-width": 2,
						},
					});

					currentMap.on("click", clusterLayerId, (event) => {
						const feature = event.features?.[0];
						if (!feature) return;
						const coordinates = feature.geometry.coordinates as [number, number];
						const clusterId = feature.properties?.cluster_id;
						if (clusterId === undefined) return;
						const source = currentMap.getSource(cameraSourceId);
						if (source instanceof GeoJSONSource) {
							void source.getClusterExpansionZoom(clusterId).then((zoom) => {
								currentMap.easeTo({ center: coordinates, zoom });
							});
						}
					});
					currentMap.on("click", cameraLayerId, (event) => {
						const feature = event.features?.[0];
						if (!feature) return;
						const properties = feature.properties as {
							id: string;
						};
						onCameraSelect?.();
						mapState.setActiveCameraId(properties.id);
					});
					currentMap.on(
						"mouseenter",
						clusterLayerId,
						() => (currentMap.getCanvas().style.cursor = "pointer"),
					);
					currentMap.on(
						"mouseleave",
						clusterLayerId,
						() => (currentMap.getCanvas().style.cursor = ""),
					);
					currentMap.on(
						"mouseenter",
						cameraLayerId,
						() => (currentMap.getCanvas().style.cursor = "pointer"),
					);
					currentMap.on(
						"mouseleave",
						cameraLayerId,
						() => (currentMap.getCanvas().style.cursor = ""),
					);
					mapState.setLoading(false);
				});
				currentMap.on("moveend", () => {
					const center = currentMap.getCenter();
					mapState.setPosition([center.lng, center.lat]);
					mapState.setZoom(currentMap.getZoom());
				});
				currentMap.on("error", (event) => {
					const message = event.error?.message ?? "MapLibre could not render the map";
					console.error("MapLibre error", event.error);
					if (mapState.loading) mapState.setError(new Error(message));
				});
			})
			.catch((error: unknown) => {
				console.error("Map style failed to load", error);
				mapState.setLoading(false);
				mapState.setError(error);
			});

		return () => {
			cancelled = true;
			mapState.setLoading(true);
			map?.remove();
		};
	});
</script>

<div class="relative h-full min-h-96 w-full">
	<div
		bind:this={mapState.container}
		class="h-full w-full"
		role="application"
		aria-label="Interactive map of Atlanta"
	></div>

	<div class="absolute bottom-12 left-6 z-20 flex flex-col gap-2 sm:bottom-6">
		<MapFilter bind:filter bind:freshness bind:availability />
		<Button size="icon" class="rounded-full" aria-label="Zoom in" onclick={zoomIn}>
			<Plus />
		</Button>
		<Button size="icon" class="rounded-full" aria-label="Zoom out" onclick={zoomOut}>
			<Minus />
		</Button>
	</div>
</div>

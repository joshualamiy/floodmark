<script lang="ts">
	import { onMount } from "svelte";
	import { createQuery } from "@tanstack/svelte-query";
	import {
		Map as MapLibreMap,
		setWorkerUrl,
		GeoJSONSource,
		type StyleSpecification,
	} from "maplibre-gl";
	import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
	import "maplibre-gl/dist/maplibre-gl.css";
	import { api } from "$lib/api";
	import { useMapState } from "$lib/state/map.svelte";

	setWorkerUrl(maplibreWorkerUrl);

	const mapState = useMapState();
	let map: MapLibreMap | undefined;
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
				predictionStatus: string | null;
			};
		}>;
	};

	function cameraData(): CameraFeatureCollection {
		return {
			type: "FeatureCollection",
			features: cameras.flatMap((camera) => {
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
							predictionStatus: camera.latestImage?.predictionStatus ?? null,
							location: [camera.locationDescription, camera.roadway, camera.direction]
								.filter(Boolean)
								.join(" | "),
						},
					},
				];
			}),
		};
	}

	function updateCameraSource() {
		const currentMap = map;
		if (!currentMap) return;

		const source = currentMap.getSource(cameraSourceId);
		if (source instanceof GeoJSONSource) void source.setData(cameraData());
	}

	$effect(() => {
		if (!mapState.loading) updateCameraSource();
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
							hasFlooded: ["max", ["case", ["==", ["get", "predictionStatus"], "flooded"], 1, 0]],
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
								[">", ["coalesce", ["get", "hasFlooded"], 0], 0],
								"#dc2626",
								"#00A6AD",
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
								["==", ["get", "predictionStatus"], "flooded"],
								"#dc2626",
								"#00A6AD",
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
						const coordinates = feature.geometry.coordinates as [number, number];
						const properties = feature.properties as {
							id: string;
						};
						const targetZoom = Math.max(currentMap.getZoom(), 14);
						currentMap.once("moveend", () => {
							currentMap.zoomTo(targetZoom, { duration: 300 });
						});
						currentMap.panTo(coordinates, { duration: 400 });
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
</div>

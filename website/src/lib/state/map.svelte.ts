import { api } from "$lib/api";
import { createQuery } from "@tanstack/svelte-query";
import { getContext, setContext } from "svelte";

const MAP_STATE_KEY = "map-state";
const MAP_STATE_CONTEXT = Symbol.for(MAP_STATE_KEY);

export type MapPosition = [longitude: number, latitude: number];

export interface MapStateOptions {
	position?: MapPosition;
	zoom?: number;
}

export class MapState {
	loading = $state(true);
	position = $state<MapPosition>([-84.388, 33.749]);
	zoom = $state(10);
	error = $state<Error | undefined>();
	container = $state<HTMLDivElement | undefined>();

	camerasQuery = createQuery(() => ({
		queryKey: ["cameras"],
		queryFn: () => api().cameras.list(),
		refetchInterval: 60_000,
	}));

	activeCameraId = $state<string | null>(null);
	activeCamera = $derived(this.camerasQuery.data?.find((camera) => camera.id === this.activeCameraId));

	constructor({ position = [-84.388, 33.749], zoom = 10 }: MapStateOptions = {}) {
		this.position = position;
		this.zoom = zoom;
	}

	setContainer(container: HTMLDivElement | undefined) {
		this.container = container;
	}

	setLoading(loading: boolean) {
		this.loading = loading;
	}

	setPosition(position: MapPosition) {
		this.position = position;
	}

	setZoom(zoom: number) {
		this.zoom = zoom;
	}

	setActiveCameraId(id: string | null) {
		this.activeCameraId = id;
	}

	setError(error: unknown) {
		this.error = error instanceof Error ? error : new Error("Unable to load the map");
	}
}

export function setMapState(options?: MapStateOptions) {
	const state = new MapState(options);
	setContext(MAP_STATE_CONTEXT, state);
	return state;
}

export function useMapState() {
	return getContext<MapState>(MAP_STATE_CONTEXT);
}

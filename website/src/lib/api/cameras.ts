import type { ApiResponse } from "$lib/types/api";
import type { Camera } from "$lib/types/camera";

export type ApiFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

async function request<T>(fetcher: ApiFetch, path: string): Promise<T> {
	const response = await fetcher(path);
	let payload: ApiResponse<T>;

	try {
		payload = (await response.json()) as ApiResponse<T>;
	} catch {
		throw new Error(`API request failed with status ${response.status}`);
	}

	if (!response.ok || payload.error) {
		throw new Error(payload.error?.message ?? `API request failed with status ${response.status}`);
	}

	return payload.data as T;
}

export function api(fetcher: ApiFetch = globalThis.fetch.bind(globalThis)) {
	return {
		camera(id: string) {
			return {
				get: () => request<Camera | null>(fetcher, `/api/cameras/${encodeURIComponent(id)}/get`),
				snapshot: (heatmap = false) =>
					request<{ url: string; expiresAt: string }>(
						fetcher,
						`/api/cameras/${encodeURIComponent(id)}/snapshot?heatmap=${heatmap}`,
					),
			};
		},
		cameras: {
			list: () => request<Camera[]>(fetcher, "/api/cameras/list"),
		},
	};
}

import type { ApiResponse } from "$lib/types/api";
import type { Camera } from "$lib/types/camera";
import type { HistoryPage } from "$lib/types/history";

export type ApiFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

async function request<T>(fetcher: ApiFetch, path: string, init?: RequestInit): Promise<T> {
	const response = await fetcher(path, init);
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
		notifications: {
			subscribe: (cameraId: string, email: string) =>
				request<{ status: "subscribed" | "verification_sent" }>(
					fetcher,
					"/api/notifications/subscribe",
					{
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ cameraId, email }),
					},
				),
		},
		camera(id: string) {
			return {
				get: () => request<Camera | null>(fetcher, `/api/cameras/${encodeURIComponent(id)}/get`),
				snapshot: (heatmap = false, imageId?: string) =>
					request<{ url: string; expiresAt: string }>(
						fetcher,
						`/api/cameras/${encodeURIComponent(id)}/snapshot?heatmap=${heatmap}${imageId ? `&imageId=${encodeURIComponent(imageId)}` : ""}`,
					),
				history: (
					options: { limit?: number; cursor?: string | null; from?: string; to?: string } = {},
				) => {
					const search = new URLSearchParams();
					if (options.limit !== undefined) search.set("limit", String(options.limit));
					if (options.cursor) search.set("cursor", options.cursor);
					if (options.from) search.set("from", options.from);
					if (options.to) search.set("to", options.to);

					const query = search.toString();
					return request<HistoryPage>(
						fetcher,
						`/api/cameras/${encodeURIComponent(id)}/history${query ? `?${query}` : ""}`,
					);
				},
			};
		},
		cameras: {
			list: () => request<Camera[]>(fetcher, "/api/cameras/list"),
		},
	};
}

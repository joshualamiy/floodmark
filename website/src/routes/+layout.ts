import { browser } from "$app/environment";
import { QueryClient } from "@tanstack/svelte-query";
import type { LayoutLoad } from "./$types";

export const load: LayoutLoad = () => {
	const queryClient = new QueryClient({
		defaultOptions: {
			queries: {
				enabled: browser,
				staleTime: 15_000,
			},
		},
	});

	return { queryClient };
};

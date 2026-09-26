import { api } from "$lib/api";
import type { PageLoad } from "./$types";

export const load: PageLoad = async ({ fetch, parent }) => {
	const { queryClient } = await parent();

	await queryClient
		.query({
			queryKey: ["cameras"],
			queryFn: () => api(fetch).cameras.list(),
		})
		.catch(() => undefined);
};

<script lang="ts">
	import ArrowLeft from "@lucide/svelte/icons/arrow-left";
	import Logo from "$components/logo.svelte";

	type Endpoint = {
		path: string;
		description: string;
		parameters?: string;
		code: string;
	};

	const endpoints: Endpoint[] = [
		{
			path: "GET /api/cameras/list",
			description: "List all active cameras with their latest image and prediction.",
			code: `const response = await fetch("/api/cameras/list");
const { data } = await response.json();`,
		},
		{
			path: "GET /api/cameras/:id/get",
			description: "Get one camera by its UUID.",
			parameters: "id · UUID",
			code: `const response = await fetch(
  \`/api/cameras/\${cameraId}/get\`
);
const { data: camera } = await response.json();`,
		},
		{
			path: "GET /api/cameras/:id/snapshot",
			description: "Get a signed URL for the latest camera snapshot or heatmap.",
			parameters: "id · UUID\nimageId · UUID · optional\nheatmap · true | false",
			code: `const response = await fetch(
  \`/api/cameras/\${cameraId}/snapshot?heatmap=true\`
);
const { data } = await response.json();`,
		},
		{
			path: "GET /api/cameras/:id/history",
			description: "Get paginated image history and flood predictions.",
			parameters: "id · UUID\nlimit · 1–50 · default 20\ncursor, from, to · optional",
			code: `const response = await fetch(
  \`/api/cameras/\${cameraId}/history?limit=20\`
);
const { data: history } = await response.json();`,
		},
	];
</script>

<svelte:head>
	<title>Floodmark API</title>
	<meta
		name="description"
		content="Floodmark API reference for camera data and flood predictions."
	/>
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
	<Logo />

	<header class="mx-auto flex max-w-4xl justify-end px-6 pt-6 sm:px-8">
		<a
			href="/map"
			class="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground"
		>
			<ArrowLeft class="size-4" />
			Back to map
		</a>
	</header>

	<main class="mx-auto max-w-4xl px-6 pt-20 pb-20 sm:px-8 sm:pt-28">
		<section class="max-w-2xl">
			<p class="font-mono text-sm text-muted-foreground">/api</p>
			<h1 class="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">Floodmark API</h1>
			<p class="mt-5 text-lg leading-8 text-muted-foreground">
				Access camera locations, snapshots, flood heatmaps, and image history with simple JSON
				endpoints.
			</p>
		</section>

		<section class="mt-16" aria-labelledby="endpoints-heading">
			<h2 id="endpoints-heading" class="text-xl font-semibold">Endpoints</h2>
			<div class="mt-5 divide-y divide-border rounded-xl border border-border bg-card">
				{#each endpoints as endpoint (endpoint.path)}
					<article class="p-5 sm:p-6">
						<h3 class="font-mono text-sm font-semibold text-primary">{endpoint.path}</h3>
						<p class="mt-2 text-sm text-muted-foreground">{endpoint.description}</p>

						{#if endpoint.parameters}
							<div class="mt-4">
								<p class="text-xs font-medium tracking-wide text-muted-foreground uppercase">
									Parameters
								</p>
								<p
									class="mt-1 font-mono text-xs leading-6 whitespace-pre-line text-muted-foreground"
								>
									{endpoint.parameters}
								</p>
							</div>
						{/if}

						<pre
							class="mt-5 overflow-x-auto rounded-lg bg-muted p-4 font-mono text-xs leading-6"><code
								>{endpoint.code}</code
							></pre>
					</article>
				{/each}
			</div>
		</section>

		<section
			class="mt-12 rounded-xl border border-border bg-muted/40 p-5 sm:p-6"
			aria-labelledby="response-heading"
		>
			<h2 id="response-heading" class="font-semibold">Response format</h2>
			<pre
				class="mt-4 overflow-x-auto rounded-lg bg-background p-4 font-mono text-xs leading-6"><code
					>{`{
  "data": { ... },
  "error": null
}`}</code
				></pre>
			<p class="mt-4 text-sm leading-6 text-muted-foreground">
				Successful responses return <code class="font-mono text-foreground">data</code>. Errors
				return an <code class="font-mono text-foreground">error</code> object with a useful message.
			</p>
		</section>
	</main>
</div>

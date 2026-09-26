<script lang="ts">
	import { api } from "$lib/api";
	import type { Camera } from "$lib/types/camera";
	import { createQuery } from "@tanstack/svelte-query";
	import { Button } from "$components/ui/button";
	import CameraHistory from "$components/camera-history.svelte";
	import SquareArrowOutUpRight from "@lucide/svelte/icons/square-arrow-out-up-right";
	import ArrowLeft from "@lucide/svelte/icons/arrow-left";
	import MapPin from "@lucide/svelte/icons/map-pin";
	import Radio from "@lucide/svelte/icons/radio";
	import { useMapState } from "$lib/state/map.svelte";
	import { Badge } from "$components/ui/badge";
	import { PredictionStatus } from "$lib/types/image-processing";
	import * as Tooltip from "$components/ui/tooltip";
	import { Spinner } from "$components/ui/spinner";
	import { Toggle } from "$components/ui/toggle";
	import MapPlus from "@lucide/svelte/icons/map-plus";
	import NotificationSignup from "$components/notification-signup.svelte";

	let { camera }: { camera: Camera } = $props();

	const map = useMapState();

	let heatmapToggle = $state(false);
	const latestImageId = $derived(camera.latestImage?.id);

	const image = createQuery(() => ({
		queryKey: ["cameras", camera.id, latestImageId, heatmapToggle ? "heatmap" : "snapshot"],
		queryFn: () => api().camera(camera.id).snapshot(heatmapToggle, latestImageId),
		enabled: Boolean(camera.id && latestImageId),
	}));

	const formatter = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit" });

	const url = $derived(image.data?.url);
	const hasFeed = $derived(Boolean(url && !url.includes("/skipped")));
	const isFlooded = $derived(camera.latestImage?.alertStatus === PredictionStatus.Flooded);
	const isWet = $derived(camera.latestImage?.alertStatus === PredictionStatus.Wet);
	const location = $derived(camera.locationDescription);

	const minutesSinceCapture = $derived(
		camera.latestImage?.capturedAt
			? Math.floor((Date.now() - new Date(camera.latestImage.capturedAt).getTime()) / 60000)
			: null,
	);

	const capturedAt = $derived(
		camera.latestImage?.capturedAt
			? formatter.format(new Date(camera.latestImage.capturedAt))
			: null,
	);

	function formatCaptureAge(minutes: number | null): string {
		if (minutes === null || minutes < 0) return "Unknown";

		if (minutes < 60) {
			return `${minutes}m`;
		}

		const hours = Math.floor(minutes / 60);

		if (hours < 24) {
			return `${hours}h`;
		}

		const days = Math.floor(hours / 24);
		return `${days}d`;
	}
</script>

{#snippet heatmap()}
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<Toggle
					bind:pressed={heatmapToggle}
					class="absolute right-3 bottom-3 cursor-pointer rounded-full border-white/80 bg-white/90 p-0 text-slate-600 shadow-lg shadow-slate-950/15 backdrop-blur-md transition hover:bg-white hover:text-accent-foreground data-[state=on]:bg-accent-foreground data-[state=on]:text-white"
					{...props}
				>
					<MapPlus class="size-4" />
				</Toggle>
			{/snippet}
		</Tooltip.Trigger>
		<Tooltip.Content side="bottom">Switch to heat map view.</Tooltip.Content>
	</Tooltip.Root>
{/snippet}

{#snippet captureStatus()}
	{#if minutesSinceCapture === null}
		<Badge variant="secondary">No data</Badge>
	{:else if minutesSinceCapture < 20}
		<Badge variant="success">
			<Radio /> Live
		</Badge>
	{:else}
		{@const age = formatCaptureAge(minutesSinceCapture)}
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Badge variant="warning">{age} ago</Badge>
			</Tooltip.Trigger>
			<Tooltip.Content>Last processed {age} ago.</Tooltip.Content>
		</Tooltip.Root>
	{/if}
{/snippet}

{#snippet predictionStatus()}
	{#if !camera.latestImage?.alertStatus}
		<Badge variant="secondary" class="p-3">No data</Badge>
	{:else if isFlooded}
		<Badge variant="destructive" class="p-3">Flood detected</Badge>
	{:else if isWet}
		<Badge variant="warning" class="p-3">Possible flooding</Badge>
	{:else}
		<Badge variant="success" class="p-3">Clear</Badge>
	{/if}
{/snippet}

<div class="flex h-full flex-col justify-between">
	<div>
		<div class="flex items-center justify-between border-b px-4 py-3 sm:px-5">
			<Button
				aria-label="Back to all cameras"
				onclick={() => map.setActiveCameraId(null)}
				size="icon-sm"
				variant="secondary"
				class="rounded-xl"
			>
				<ArrowLeft />
			</Button>
			{@render captureStatus()}
		</div>
		<div class="flex flex-1 flex-col justify-between px-4 py-3 sm:px-5">
			<div
				class="relative mx-auto aspect-5/3 w-4/5 overflow-hidden rounded-2xl bg-slate-100 shadow-inner ring-1 ring-accent-foreground/10 sm:w-full"
			>
				{#if hasFeed}
					<img
						src={url}
						alt={`Latest snapshot from ${camera.name}`}
						class="h-full w-full object-contain"
					/>
					{@render heatmap()}
					<Badge class="absolute bottom-3 left-3 p-3 text-xs">
						{capturedAt ? `Captured ${capturedAt}` : "Awaiting snapshot"}
					</Badge>
				{:else if image.isPending}
					<div class="flex h-full flex-col items-center justify-center gap-3 text-slate-400">
						<Spinner class="size-6" />
						<p class="text-xs font-medium">Loading camera feed</p>
					</div>
					{@render heatmap()}
				{:else}
					<div class="flex h-full flex-col items-center justify-center gap-2 text-slate-400">
						<Radio class="size-7" strokeWidth={1.5} />
						<p class="text-xs font-medium">No camera feed available</p>
					</div>
				{/if}
			</div>
			<div class="mt-5 flex items-start justify-between gap-3">
				<div class="min-w-0">
					<h1 class="truncate text-xl font-semibold">{camera.name}</h1>
					{#if location}
						<p class="mt-1 flex items-start gap-1.5 text-xs leading-5 text-slate-500">
							<MapPin class="mt-0.5 size-3.5 shrink-0" />
							<span>{location}</span>
						</p>
					{/if}
				</div>
				{@render predictionStatus()}
			</div>
			{#if camera.latestImage?.alertNote || camera.latestImage?.predictionNote}
				<p class="mt-2 text-xs text-muted-foreground">
					{camera.latestImage.alertNote ?? camera.latestImage.predictionNote}
				</p>
			{/if}
			{#if camera.latestImage?.heatmapNote}
				<p class="mt-1 text-xs text-muted-foreground">{camera.latestImage.heatmapNote}</p>
			{/if}
			<CameraHistory cameraId={camera.id} cameraName={camera.name} />
		</div>
	</div>

	<div class="m-4 flex flex-col gap-2">
		<NotificationSignup {camera} />
		{#if camera.sourceUrl}
			<Button
				class="h-10 justify-between rounded-xl bg-accent-foreground px-4 text-white shadow-sm hover:bg-accent-foreground/90"
				href={camera.sourceUrl}
				target="_blank"
				rel="noreferrer"
			>
				<span>Open source (511ga.org)</span>
				<SquareArrowOutUpRight class="size-4" />
			</Button>
		{/if}
	</div>
</div>

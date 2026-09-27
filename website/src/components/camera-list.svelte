<script lang="ts">
	import Search from "@lucide/svelte/icons/search";
	import MapPin from "@lucide/svelte/icons/map-pin";
	import { Badge } from "$components/ui/badge";
	import { Button } from "$components/ui/button";
	import type { Camera } from "$lib/types/camera";
	import { PredictionStatus } from "$lib/types/image-processing";
	import { useMapState } from "$lib/state/map.svelte";
	import { ScrollArea } from "$components/ui/scroll-area";

	type Filter = "all" | "clear" | "wet" | "flooded";
	const ROW_HEIGHT = 120;
	const OVERSCAN = 4;

	let { onCameraSelect }: { onCameraSelect?: () => void } = $props();

	const map = useMapState();
	let search = $state("");
	let filter = $state<Filter>("all");
	let scrollTop = $state(0);
	let viewportHeight = $state(0);
	let scrollViewport = $state<HTMLElement | null>(null);

	$effect(() => {
		const viewport = scrollViewport;
		if (!viewport) return;

		const updateScrollTop = () => {
			scrollTop = viewport.scrollTop;
		};

		viewport.addEventListener("scroll", updateScrollTop);
		return () => viewport.removeEventListener("scroll", updateScrollTop);
	});

	const cameras = $derived(map.camerasQuery.data ?? []);
	const filteredCameras = $derived.by(() => {
		const term = search.trim().toLowerCase();

		return cameras.filter((camera) => {
			const searchableText = [
				camera.name,
				camera.locationDescription,
				camera.roadway,
				camera.direction,
			]
				.filter(Boolean)
				.join(" ")
				.toLowerCase();
			const status = getStatus(camera);
			const matchesFilter =
				filter === "all" ||
				(filter === "clear"
					? camera.latestImage?.alertStatus === PredictionStatus.Dry
					: status === filter);

			return (!term || searchableText.includes(term)) && matchesFilter;
		});
	});
	const visibleCount = $derived(Math.ceil(viewportHeight / ROW_HEIGHT) + OVERSCAN * 2);
	const visibleStart = $derived(
		Math.min(
			Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN),
			Math.max(0, filteredCameras.length - visibleCount),
		),
	);
	const visibleCameras = $derived(filteredCameras.slice(visibleStart, visibleStart + visibleCount));
	const bottomSpacerHeight = $derived(
		Math.max(0, filteredCameras.length - visibleStart - visibleCameras.length) * ROW_HEIGHT,
	);

	const filters: Array<{ value: Filter; label: string }> = [
		{ value: "all", label: "All" },
		{ value: "flooded", label: "Flooded" },
		{ value: "wet", label: "Wet" },
		{ value: "clear", label: "Clear" },
	];

	function getStatus(camera: Camera): Exclude<Filter, "all"> {
		if (camera.latestImage?.alertStatus === PredictionStatus.Flooded) return "flooded";
		if (camera.latestImage?.alertStatus === PredictionStatus.Wet) return "wet";
		return "clear";
	}

	function statusLabel(camera: Camera): string {
		if (!camera.latestImage?.alertStatus) return "No feed";
		if (getStatus(camera) === "wet") return "Possible flooding";
		return getStatus(camera)[0].toUpperCase() + getStatus(camera).slice(1);
	}

	function statusVariant(camera: Camera): "destructive" | "warning" | "success" | "secondary" {
		const status = getStatus(camera);
		if (status === "flooded") return "destructive";
		if (status === "wet") return "warning";
		if (!camera.latestImage?.alertStatus) return "secondary";
		return "success";
	}

	function location(camera: Camera): string {
		return [camera.locationDescription, camera.roadway, camera.direction]
			.filter(Boolean)
			.join(" | ");
	}
</script>

<div class="flex h-full min-h-0 flex-1 flex-col">
	<div class="border-b px-4 py-4 sm:px-5">
		<div class="mb-4 flex items-start justify-between gap-3">
			<div>
				<p class="text-xs font-semibold tracking-[0.16em] text-muted-foreground uppercase">
					Atlanta
				</p>
				<h1 class="mt-1 text-xl font-semibold tracking-tight">Traffic cameras</h1>
			</div>
			<p class="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
				{filteredCameras.length} of {cameras.length}
			</p>
		</div>

		<label class="relative block">
			<span class="sr-only">Search cameras</span>
			<Search
				class="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
			/>
			<input
				bind:value={search}
				class="h-10 w-full rounded-xl border border-input bg-background pr-3 pl-9 text-sm transition outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
				placeholder="Search by name or location"
				type="search"
			/>
		</label>

		<div class="mt-3 flex gap-1.5 overflow-x-auto pb-1" aria-label="Filter cameras">
			{#each filters as option (option.value)}
				<Button
					aria-pressed={filter === option.value}
					class="shrink-0 rounded-full"
					onclick={() => (filter = option.value)}
					size="sm"
					variant={filter === option.value ? "secondary" : "ghost"}
				>
					{option.label}
				</Button>
			{/each}
		</div>
	</div>

	<div bind:clientHeight={viewportHeight} class="min-h-0 flex-1 py-3 pl-3 sm:pl-4">
		<ScrollArea
			bind:viewportRef={scrollViewport}
			class="size-full"
			data-vaul-no-drag
			scrollbarYClasses="w-2.5"
			viewportClass="scroll-fade touch-pan-y pr-3 sm:pr-4"
		>
			{#if map.camerasQuery.isPending}
				<div
					class="flex h-full items-center justify-center px-4 py-8 text-sm text-muted-foreground"
				>
					Loading cameras...
				</div>
			{:else if map.camerasQuery.isError}
				<div
					class="flex h-full items-center justify-center px-6 py-8 text-center text-sm text-destructive"
				>
					Unable to load cameras.
				</div>
			{:else if filteredCameras.length === 0}
				<div class="flex h-full flex-col items-center justify-center px-6 py-8 text-center">
					<p class="text-sm font-medium">No cameras found</p>
					<p class="mt-1 text-xs text-muted-foreground">Try a different search or filter.</p>
				</div>
			{:else}
				<div>
					<div style:height={`${visibleStart * ROW_HEIGHT}px`}></div>
					{#each visibleCameras as camera (camera.id)}
						<button
							class="group mb-2 h-22 w-full cursor-pointer rounded-2xl border border-transparent bg-muted/50 px-3 py-3 text-left transition hover:border-border hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
							onclick={() => {
								onCameraSelect?.();
								map.setActiveCameraId(camera.id);
							}}
							type="button"
						>
							<div class="flex items-start justify-between gap-3">
								<div class="min-w-0">
									<p class="truncate text-sm font-semibold">{camera.name}</p>
									{#if location(camera)}
										<p
											class="mt-1 flex items-start gap-1.5 text-xs leading-4 text-muted-foreground"
										>
											<MapPin class="mt-0.5 size-3.5 shrink-0" />
											<span class="line-clamp-2">{location(camera)}</span>
										</p>
									{/if}
								</div>
								<Badge variant={statusVariant(camera)}>{statusLabel(camera)}</Badge>
							</div>
							<!-- <div class="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
							<Radio class="size-3.5" />
							{camera.latestImage?.capturedAt ? "Recent capture available" : "Awaiting capture"}
						</div> -->
						</button>
					{/each}
					<div style:height={`${bottomSpacerHeight}px`}></div>
				</div>
			{/if}
		</ScrollArea>
	</div>
</div>

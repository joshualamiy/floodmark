<script lang="ts" module>
	export type MapFilterOption = "all" | "clear" | "wet" | "flooded" | "no-data";
	export type MapFreshnessFilter = "all" | "recent" | "stale" | "no-capture";
	export type MapAvailabilityFilter = "all" | "available" | "unavailable" | "error";
</script>

<script lang="ts">
	import Filter from "@lucide/svelte/icons/filter";
	import Button from "./ui/button/button.svelte";
	import * as Popover from "./ui/popover";
	type FilterOption = MapFilterOption;

	interface Props {
		filter?: FilterOption;
		freshness?: MapFreshnessFilter;
		availability?: MapAvailabilityFilter;
	}

	let {
		filter = $bindable<FilterOption>("all"),
		freshness = $bindable<MapFreshnessFilter>("all"),
		availability = $bindable<MapAvailabilityFilter>("all"),
	}: Props = $props();

	const filters: Array<{ value: FilterOption; label: string }> = [
		{ value: "all", label: "All" },
		{ value: "flooded", label: "Flooded" },
		{ value: "wet", label: "Wet" },
		{ value: "clear", label: "Clear" },
		{ value: "no-data", label: "No feed" },
	];
	const freshnessFilters: Array<{ value: MapFreshnessFilter; label: string }> = [
		{ value: "all", label: "Any capture age" },
		{ value: "recent", label: "Recent (<20 min)" },
		{ value: "stale", label: "Stale (20+ min)" },
		{ value: "no-capture", label: "No capture" },
	];
	const availabilityFilters: Array<{ value: MapAvailabilityFilter; label: string }> = [
		{ value: "all", label: "Any availability" },
		{ value: "available", label: "Has image" },
		{ value: "unavailable", label: "No image" },
		{ value: "error", label: "Processing error" },
	];

	const isFiltered = $derived(filter !== "all" || freshness !== "all" || availability !== "all");
</script>

<Popover.Root>
	<Popover.Trigger>
		{#snippet child({ props })}
			<Button
				{...props}
				aria-label="Filter map markers"
				aria-pressed={isFiltered}
				class="rounded-full"
				size="icon"
			>
				<Filter />
			</Button>
		{/snippet}
	</Popover.Trigger>
	<Popover.Content align="start" class="w-72" side="top">
		<Popover.Title>Filter map markers</Popover.Title>
		<div class="flex flex-col gap-4">
			<div class="flex flex-col gap-1" aria-label="Condition filters" role="group">
				<p class="px-2 text-xs font-medium text-muted-foreground">Condition</p>
				<div class="grid grid-cols-5 gap-1">
					{#each filters as option (option.value)}
						<Button
							aria-pressed={filter === option.value}
							class="min-w-0 px-1 text-xs"
							onclick={() => (filter = option.value)}
							size="sm"
							variant={filter === option.value ? "secondary" : "ghost"}
						>
							{option.label}
						</Button>
					{/each}
				</div>
			</div>
			<label
				class="flex flex-col gap-1.5 text-xs font-medium text-muted-foreground"
				for="map-freshness"
			>
				Capture freshness
				<select
					bind:value={freshness}
					class="h-9 rounded-md border border-input bg-background px-2.5 text-sm font-normal text-foreground outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
					id="map-freshness"
				>
					{#each freshnessFilters as option (option.value)}
						<option value={option.value}>{option.label}</option>
					{/each}
				</select>
			</label>
			<label
				class="flex flex-col gap-1.5 text-xs font-medium text-muted-foreground"
				for="map-availability"
			>
				Availability
				<select
					bind:value={availability}
					class="h-9 rounded-md border border-input bg-background px-2.5 text-sm font-normal text-foreground outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
					id="map-availability"
				>
					{#each availabilityFilters as option (option.value)}
						<option value={option.value}>{option.label}</option>
					{/each}
				</select>
			</label>
		</div>
	</Popover.Content>
</Popover.Root>

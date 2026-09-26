<script lang="ts">
	import BellRing from "@lucide/svelte/icons/bell-ring";
	import { api } from "$lib/api";
	import type { Camera } from "$lib/types/camera";
	import { Button } from "$components/ui/button";
	import * as Dialog from "$components/ui/dialog";
	import * as Item from "$components/ui/item";
	import { toast } from "svelte-sonner";

	let { camera }: { camera: Camera } = $props();
	let email = $state("");
	let formState = $state<"idle" | "submitting" | "success" | "error">("idle");

	async function submit() {
		formState = "submitting";
		try {
			const result = await api().notifications.subscribe(camera.id, email);
			formState = "success";
			toast.success(
				result.status === "verification_sent"
					? "Check your inbox to verify this email."
					: "Alerts enabled for this camera.",
			);
		} catch (error) {
			formState = "error";
			toast.error(error instanceof Error ? error.message : "Unable to enable alerts.");
		}
	}
</script>

<Dialog.Root>
	<Item.Root size="sm" variant="muted" class="bg-accent rounded-xl">
		<Item.Media variant="icon">
			<BellRing />
		</Item.Media>
		<Item.Content>
			<Item.Title>Get flood alerts</Item.Title>
		</Item.Content>
		<Item.Actions>
			<Dialog.Trigger>
				{#snippet child({ props })}
					<Button size="sm" {...props}>Sign up</Button>
				{/snippet}
			</Dialog.Trigger>
		</Item.Actions>
	</Item.Root>

	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Get flood alerts</Dialog.Title>
			<Dialog.Description>
				You will need to verify your email, and an admin must approve your request before you can
				receive notifications.
			</Dialog.Description>
		</Dialog.Header>

		<form
			class="flex flex-col gap-3"
			onsubmit={(event) => {
				event.preventDefault();
				submit();
			}}
		>
			<label class="text-sm font-medium text-slate-700" for="notification-email"
				>Email address</label
			>
			<div class="flex gap-2">
				<input
					id="notification-email"
					bind:value={email}
					type="email"
					required
					placeholder="you@example.com"
					autocomplete="email"
					disabled={formState === "submitting"}
					class="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm ring-accent-foreground/20 outline-none placeholder:text-slate-400 focus:ring-3 disabled:opacity-60"
				/>
				<Button type="submit" size="sm" disabled={formState === "submitting"}>
					{formState === "submitting" ? "Sending" : "Notify me"}
				</Button>
			</div>
		</form>
	</Dialog.Content>
</Dialog.Root>

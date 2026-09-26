import { and, eq, gt } from "drizzle-orm";
import { redirect } from "@sveltejs/kit";
import { getDB } from "$lib/server/database";
import { notificationEmails } from "$lib/server/schema";
import { hashToken } from "$lib/server/notification-tokens";

export async function GET({ url }) {
	const token = url.searchParams.get("token");
	if (token) {
		await getDB()
			.update(notificationEmails)
			.set({
				verifiedAt: new Date(),
				verificationTokenHash: null,
				verificationTokenExpiresAt: null,
				updatedAt: new Date(),
			})
			.where(
				and(
					eq(notificationEmails.verificationTokenHash, hashToken(token)),
					gt(notificationEmails.verificationTokenExpiresAt, new Date()),
				),
			);
	}

	redirect(303, "/map?notification=verified");
}

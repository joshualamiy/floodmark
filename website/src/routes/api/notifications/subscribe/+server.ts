import { eq } from "drizzle-orm";
import { error } from "@sveltejs/kit";
import { z } from "zod";
import { env } from "$env/dynamic/private";
import { api } from "$lib/server/api";
import { getDB } from "$lib/server/database";
import { cameraNotificationEmails, cameras, notificationEmails } from "$lib/server/schema";
import { appUrl, createVerificationToken } from "$lib/server/notification-tokens";
import { sendEmail } from "$lib/server/resend";

const bodySchema = z.object({
	cameraId: z.uuid(),
	email: z.email().transform((value) => value.trim().toLowerCase()),
});
const outputSchema = z.object({ status: z.enum(["subscribed", "verification_sent"]) });

function escapeHtml(value: string): string {
	return value.replace(
		/[&<>"']/g,
		(character) =>
			({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!,
	);
}

export const POST = api({
	id: "notifications.subscribe",
	name: "Subscribe to camera alerts",
	description: "Subscribe an email address to flood alerts for a camera",
	schema: { body: bodySchema, output: outputSchema },
	handle: async ({ body }) => {
		const db = getDB();
		const [camera] = await db
			.select({ id: cameras.id, name: cameras.name })
			.from(cameras)
			.where(eq(cameras.id, body.cameraId))
			.limit(1);
		if (!camera) error(404, "Camera not found");

		const [existing] = await db
			.select()
			.from(notificationEmails)
			.where(eq(notificationEmails.email, body.email))
			.limit(1);
		const email =
			existing ??
			(await db.insert(notificationEmails).values({ email: body.email }).returning())[0];
		if (!email) throw new Error("Unable to create notification email");

		await db
			.insert(cameraNotificationEmails)
			.values({ cameraId: body.cameraId, emailId: email.id })
			.onConflictDoNothing();

		if (email.verifiedAt) return { status: "subscribed" as const };

		const verification = createVerificationToken();
		await db
			.update(notificationEmails)
			.set({
				verificationTokenHash: verification.hash,
				verificationTokenExpiresAt: verification.expiresAt,
				updatedAt: new Date(),
			})
			.where(eq(notificationEmails.id, email.id));

		await sendEmail({
			from: env.RESEND_VERIFICATION_FROM ?? "verify@floodmark.tech",
			to: email.email,
			subject: "Verify your Floodmark flood alerts",
			html: `<p>Confirm flood alerts for <strong>${escapeHtml(camera.name)}</strong> by clicking the link below.</p><p><a href="${appUrl()}/api/notifications/verify?token=${verification.token}">Verify email</a></p><p>This link expires in 24 hours.</p>`,
		});

		return { status: "verification_sent" as const };
	},
});

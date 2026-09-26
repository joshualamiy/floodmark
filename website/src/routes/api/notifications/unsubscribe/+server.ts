import { and, eq } from "drizzle-orm";
import { redirect } from "@sveltejs/kit";
import { getDB } from "$lib/server/database";
import { cameraNotificationEmails } from "$lib/server/schema";
import { isValidUnsubscribeToken } from "$lib/server/notification-tokens";

export async function GET({ url }) {
	const emailId = url.searchParams.get("email");
	const cameraId = url.searchParams.get("camera");
	const token = url.searchParams.get("token");
	if (emailId && cameraId && token && isValidUnsubscribeToken(token, emailId, cameraId)) {
		await getDB()
			.delete(cameraNotificationEmails)
			.where(
				and(
					eq(cameraNotificationEmails.emailId, emailId),
					eq(cameraNotificationEmails.cameraId, cameraId),
				),
			);
	}

	redirect(303, "/map?notification=unsubscribed");
}

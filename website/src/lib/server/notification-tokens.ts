import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { env } from "$env/dynamic/private";

const TOKEN_TTL_MS = 24 * 60 * 60 * 1000;

function secret(): string {
	if (!env.NOTIFICATION_UNSUBSCRIBE_SECRET)
		throw new Error("NOTIFICATION_UNSUBSCRIBE_SECRET is not configured");
	return env.NOTIFICATION_UNSUBSCRIBE_SECRET;
}

export function createVerificationToken(): { token: string; hash: string; expiresAt: Date } {
	const token = randomBytes(32).toString("hex");
	return {
		token,
		hash: hashToken(token),
		expiresAt: new Date(Date.now() + TOKEN_TTL_MS),
	};
}

export function hashToken(token: string): string {
	return createHash("sha256").update(token).digest("hex");
}

export function createUnsubscribeToken(emailId: string, cameraId: string): string {
	return createHmac("sha256", secret()).update(`${emailId}:${cameraId}`).digest("hex");
}

export function isValidUnsubscribeToken(token: string, emailId: string, cameraId: string): boolean {
	const expected = Buffer.from(createUnsubscribeToken(emailId, cameraId), "hex");
	const supplied = Buffer.from(token, "hex");
	return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

export function appUrl(): string {
	return (env.APP_URL ?? "http://localhost:5173").replace(/\/$/, "");
}

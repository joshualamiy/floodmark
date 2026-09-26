import { env } from "$env/dynamic/private";

type EmailMessage = {
	from: string;
	to: string;
	subject: string;
	html: string;
};

export async function sendEmail(message: EmailMessage): Promise<void> {
	if (!env.RESEND_API_KEY) throw new Error("RESEND_API_KEY is not configured");

	const response = await fetch("https://api.resend.com/emails", {
		method: "POST",
		headers: {
			Authorization: `Bearer ${env.RESEND_API_KEY}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(message),
	});

	if (!response.ok) {
		throw new Error(`Resend request failed with status ${response.status}`);
	}
}

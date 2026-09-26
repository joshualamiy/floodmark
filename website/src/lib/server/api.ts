import { isHttpError, json } from "@sveltejs/kit";
import type { RequestHandler } from "@sveltejs/kit";
import { z } from "zod";
import type { ApiError, ApiOptions, ApiResponse, ApiSchema, ParsedValue } from "$lib/types/api";

function serializeError(error: unknown): ApiError {
	if (error instanceof z.ZodError) {
		return {
			name: error.name,
			message: error.message,
			issues: error.issues,
		};
	}

	if (error instanceof Error) {
		return {
			name: error.name,
			message: error.message,
		};
	}

	return {
		name: "Error",
		message: typeof error === "string" ? error : "An unexpected error occurred",
	};
}

function failure<T>(error: unknown, status: number) {
	return json({ data: null, error: serializeError(error) } satisfies ApiResponse<T>, { status });
}

export function api<S extends ApiSchema>({ schema, handle }: ApiOptions<S>): RequestHandler {
	return async ({ request, url, params }) => {
		let body: unknown = null;

		if (schema.body) {
			try {
				body = await request.json();
			} catch (error) {
				return failure(error, 400);
			}
		}

		const parsedBody = schema.body?.safeParse(body);
		if (parsedBody && !parsedBody.success) return failure(parsedBody.error, 400);

		const parsedQuery = schema.query?.safeParse(Object.fromEntries(url.searchParams));
		if (parsedQuery && !parsedQuery.success) return failure(parsedQuery.error, 400);

		const parsedParams = schema.params?.safeParse(params);
		if (parsedParams && !parsedParams.success) return failure(parsedParams.error, 400);

		try {
			const result = await handle({
				body: (parsedBody?.data ?? null) as ParsedValue<S["body"]>,
				query: (parsedQuery?.data ?? null) as ParsedValue<S["query"]>,
				params: (parsedParams?.data ?? null) as ParsedValue<S["params"]>,
			});
			const output = schema.output.safeParse(result);

			if (!output.success) return failure(output.error, 500);

			return json({
				data: output.data as z.output<S["output"]>,
				error: null,
			} satisfies ApiResponse<z.output<S["output"]>>);
		} catch (error) {
			if (isHttpError(error)) return failure(new Error(error.body.message), error.status);
			return failure(error, 500);
		}
	};
}

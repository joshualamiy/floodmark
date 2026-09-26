import { z } from "zod";

export type Schema = z.ZodType;

export type ApiSchema = {
	body?: Schema;
	query?: Schema;
	params?: Schema;
	output: Schema;
};

export type ParsedValue<T extends Schema | undefined> = T extends Schema ? z.output<T> : null;

export type ApiInput<S extends ApiSchema> = {
	body: ParsedValue<S["body"]>;
	query: ParsedValue<S["query"]>;
	params: ParsedValue<S["params"]>;
};

export type ApiError = {
	name: string;
	message: string;
	issues?: z.core.$ZodIssue[];
};

export type ApiResponse<T> = {
	data: T | null;
	error: ApiError | null;
};

export type ApiOptions<S extends ApiSchema> = {
	id: string;
	name: string;
	description: string;
	schema: S;
	handle: (input: ApiInput<S>) => z.output<S["output"]> | Promise<z.output<S["output"]>>;
};

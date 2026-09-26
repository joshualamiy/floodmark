import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const argumentsList = process.argv.slice(2);
if (argumentsList[0] === "--") argumentsList.shift();

const [inputArgument, outputArgument] = argumentsList;

if (!inputArgument || !outputArgument) {
	console.error("Usage: node scripts/cameras-json-to-sql.mjs <input.json> <output.sql>");
	process.exit(1);
}

const inputPath = resolve(inputArgument);
const outputPath = resolve(outputArgument);

function sqlValue(value) {
	if (value === null || value === undefined) return "NULL";
	if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
	if (typeof value === "number") {
		if (!Number.isFinite(value)) throw new Error("Cannot write a non-finite number to SQL");
		return String(value);
	}

	return `'${String(value).replaceAll("'", "''")}'`;
}

function requiredString(value, field, index) {
	if (value === null || value === undefined || value === "") {
		throw new Error(`Camera ${index} is missing ${field}`);
	}

	return String(value);
}

function requiredNumber(value, field, index) {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		throw new Error(`Camera ${index} has an invalid ${field}`);
	}

	return value;
}

const input = JSON.parse(await readFile(inputPath, "utf8"));

if (!Array.isArray(input) || input.length === 0) {
	throw new Error("Input JSON must be a non-empty array of cameras");
}

const values = input.map((camera, index) => {
	if (!camera || typeof camera !== "object") {
		throw new Error(`Camera ${index} is not an object`);
	}

	if (!Array.isArray(camera.Views) || camera.Views.length !== 1) {
		throw new Error(`Camera ${index} must contain exactly one view`);
	}

	const view = camera.Views[0];

	return [
		requiredString(camera.Source, "Source", index),
		requiredString(camera.Id, "Id", index),
		requiredString(camera.SourceId, "SourceId", index),
		requiredString(camera.Name, "Name", index),
		camera.Roadway ?? null,
		camera.Direction ?? null,
		camera.Location ?? null,
		requiredNumber(camera.Latitude, "Latitude", index),
		requiredNumber(camera.Longitude, "Longitude", index),
		camera.SortOrder ?? null,
		requiredString(view.Id, "Views[0].Id", index),
		view.Url ?? null,
		view.Status ?? null,
		view.Description ?? null,
		view.SortId ?? null,
		true,
	].map(sqlValue);
});

const columns = [
	"source",
	"source_camera_id",
	"source_id",
	"name",
	"roadway",
	"direction",
	"location_description",
	"latitude",
	"longitude",
	"sort_order",
	"source_view_id",
	"source_url",
	"source_view_status",
	"source_view_description",
	"source_view_sort_id",
	"is_active",
].map((column) => `"${column}"`);

const sql = [
	`INSERT INTO "cameras" (${columns.join(", ")})`,
	"VALUES",
	values.map((row) => `\t(${row.join(", ")})`).join(",\n"),
	";",
	"",
].join("\n");

await writeFile(outputPath, sql, "utf8");
console.log(`Wrote ${values.length} camera rows to ${outputPath}`);

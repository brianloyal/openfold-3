## Production images

TODO

## Development images

These images are the biggest but come with all the build tooling, needed to compile things at runtime (Deepspeed)

```
docker build -f docker/development/Dockerfile -t openfold-docker:latest .
```

For Blackwell image build, see [Build_instructions_blackwell.md](Build_instructions_blackwell.md)

## Test images

Build the test image
```
docker build -f docker/test/Dockerfile -t openfold-docker:tests .
```

Run the unit tests
```
docker run --rm -v $(pwd -P):/opt/openfold3 -t openfold-docker:tests pytest openfold3/tests -vvv
```

#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPosition;
out vec3 vNormal;

void main()
{
    vec4 worldPosition =
        uModel * vec4(aPosition, 1.0);

    vWorldPosition =
        worldPosition.xyz;

    mat3 normalMatrix =
        mat3(
            transpose(
                inverse(uModel)
            )
        );

    vNormal =
        normalize(
            normalMatrix * aNormal
        );

    gl_Position =
        uProjection *
        uView *
        worldPosition;
}
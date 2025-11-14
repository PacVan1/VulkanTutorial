#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // force depth range of 0.0 - 1.0
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobj/tiny_obj_loader.h>

#include <vulkan/vulkan.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <array>
#include <set>
#include <optional>
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <fstream>

#ifdef max
#undef max
#endif

static constexpr uint32_t kWindowWidth			= 800;
static constexpr uint32_t kWindowHeight			= 600;
static constexpr uint32_t kMaxFramesInFlight	= 2;

#if defined (_DEBUG)
static constexpr bool kEnableVkValidationLayers = true;
#else
static constexpr bool kEnableVkValidationLayers = false;
#endif

static const std::vector<const char*> g_vkValidationLayers	 = { "VK_LAYER_KHRONOS_validation" };
static const std::vector<const char*> g_physDeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

static const std::string kModelPath		= "models/viking_room.obj";
static const std::string kTexturePath	= "textures/viking_room.png";

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

// Proxy functions
static VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance, 
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
	const VkAllocationCallbacks* pAllocator, 
	VkDebugUtilsMessengerEXT* pDebugMessenger) 
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) 
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else 
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

static void DestroyDebugUtilsMessengerEXT(
	VkInstance instance,
	VkDebugUtilsMessengerEXT debugMessenger, 
	const VkAllocationCallbacks* pAllocator) 
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) 
	{
		func(instance, debugMessenger, pAllocator);
	}
}

static std::vector<char> ReadFile(const std::string& filename) 
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) 
	{
		throw std::runtime_error("Failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);

struct Vertex
{
	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding		= 0;
		bindingDescription.stride		= sizeof(Vertex);
		bindingDescription.inputRate	= VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions() 
	{
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
		attributeDescriptions[0].binding	= 0;
		attributeDescriptions[0].location	= 0;
		attributeDescriptions[0].format		= VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset		= offsetof(Vertex, pos);
		attributeDescriptions[1].binding	= 0;
		attributeDescriptions[1].location	= 1;
		attributeDescriptions[1].format		= VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset		= offsetof(Vertex, color);
		attributeDescriptions[2].binding	= 0;
		attributeDescriptions[2].location	= 2;
		attributeDescriptions[2].format		= VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset		= offsetof(Vertex, texCoord);
		return attributeDescriptions;
	}

	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct QueueFamilyIndices 
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool IsComplete() const
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails 
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class Application
{
public:
	void Run()
	{
		InitWindow();
		InitVk();
		MainLoop();
		Cleanup();
	}

private:
	void InitWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		m_window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Vulkan", nullptr, nullptr);

		glfwSetWindowUserPointer(m_window, this);
		glfwSetFramebufferSizeCallback(m_window, FramebufferResizeCallback);
	}

	void InitVk()
	{
		CreateVkInstance();
		SetupVkDebugMessenger();
		CreateVkSurface();
		PickPhysicalDevice();
		CreateLogicalDevice();
		CreateSwapChain();
		CreateImageViews();
		CreateRenderPass();
		CreateDescriptorSetLayout();
		CreateGraphicsPipeline();
		CreateCommandPool();
		CreateDepthResources();
		CreateFramebuffers();
		CreateTextureImage();
		CreateTextureImageView();
		CreateTextureSampler();
		LoadModel();
		CreateVertexBuffer();
		CreateIndexBuffer();
		CreateUniformBuffers();
		CreateDescriptorPool();
		CreateDescriptorSets();
		CreateCommandBuffer();
		CreateSyncObjects();
	}

	void LoadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		std::string warn;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, kModelPath.c_str())) 
		{
			throw std::runtime_error(err);
		}

		for (const auto& shape : shapes) 
		{
			for (const auto& index : shape.mesh.indices) 
			{
				Vertex vertex{};

				vertex.pos = 
				{
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.texCoord = 
				{
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				m_vertices.push_back(vertex);
				m_indices.push_back(m_indices.size());
			}
		}
	}

	bool HasStencilComponent(VkFormat format) 
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	VkFormat FindDepthFormat() 
	{
		return FindSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates) 
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) 
			{
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) 
			{
				return format;
			}
		}

		throw std::runtime_error("Failed to find supported format!");
	}

	void CreateDepthResources()
	{
		VkFormat depthFormat = FindDepthFormat();
		CreateImage(m_vkSwapChainExtent.width, m_vkSwapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthImage, m_depthImageMemory);
		m_depthImageView = CreateImageView(m_depthImage, depthFormat, VK_IMAGE_ASPECT_COLOR_BIT);

		TransitionImageLayout(m_depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	void CreateTextureSampler()
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType					= VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter				= VK_FILTER_LINEAR;
		samplerInfo.minFilter				= VK_FILTER_LINEAR;
		samplerInfo.addressModeU			= VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV			= VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW			= VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable		= VK_TRUE;
		samplerInfo.borderColor				= VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable			= VK_FALSE;
		samplerInfo.compareOp				= VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode				= VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias				= 0.0f;
		samplerInfo.minLod					= 0.0f;
		samplerInfo.maxLod					= 0.0f;

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &properties);
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

		if (vkCreateSampler(m_vkDevice, &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create texture sampler!");
		}
	}

	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType								= VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image								= image;
		viewInfo.viewType							= VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format								= format;
		viewInfo.subresourceRange.aspectMask		= aspectFlags;
		viewInfo.subresourceRange.baseMipLevel		= 0;
		viewInfo.subresourceRange.levelCount		= 1;
		viewInfo.subresourceRange.baseArrayLayer	= 0;
		viewInfo.subresourceRange.layerCount		= 1;

		VkImageView imageView;
		if (vkCreateImageView(m_vkDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create image view!");
		}

		return imageView;
	}

	void CreateTextureImageView()
	{
		m_textureImageView = CreateImageView(m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		EndSingleTimeCommands(commandBuffer);
	}

	void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) 
	{
		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) 
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (HasStencilComponent(format)) 
			{
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else 
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) 
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) 
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) 
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else 
		{
			throw std::invalid_argument("Unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		EndSingleTimeCommands(commandBuffer);
	}

	VkCommandBuffer BeginSingleTimeCommands() 
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_vkCommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(m_vkDevice, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void EndSingleTimeCommands(VkCommandBuffer commandBuffer) 
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(m_vkGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(m_vkGraphicsQueue);

		vkFreeCommandBuffers(m_vkDevice, m_vkCommandPool, 1, &commandBuffer);
	}

	void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) 
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType			= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType		= VK_IMAGE_TYPE_2D;
		imageInfo.extent.width	= width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth	= 1;
		imageInfo.mipLevels		= 1;
		imageInfo.arrayLayers	= 1;
		imageInfo.format		= format;
		imageInfo.tiling		= tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage			= usage;
		imageInfo.samples		= VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode	= VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(m_vkDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(m_vkDevice, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(m_vkDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to allocate image memory!");
		}

		vkBindImageMemory(m_vkDevice, image, imageMemory, 0);
	}

	void CreateTextureImage()
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(kTexturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels) 
		{
			throw std::runtime_error("Failed to load texture image!");
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(m_vkDevice, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(m_vkDevice, stagingBufferMemory);

		stbi_image_free(pixels);

		VkImageCreateInfo imageInfo{};
		imageInfo.sType			= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType		= VK_IMAGE_TYPE_2D;
		imageInfo.extent.width	= static_cast<uint32_t>(texWidth);
		imageInfo.extent.height = static_cast<uint32_t>(texHeight);
		imageInfo.extent.depth	= 1;
		imageInfo.mipLevels		= 1;
		imageInfo.arrayLayers	= 1;
		imageInfo.format		= VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.tiling		= VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage			= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode	= VK_SHARING_MODE_EXCLUSIVE;

		CreateImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_textureImage, m_textureImageMemory);

		TransitionImageLayout(m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		CopyBufferToImage(stagingBuffer, m_textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		TransitionImageLayout(m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(m_vkDevice, stagingBuffer, nullptr);
		vkFreeMemory(m_vkDevice, stagingBufferMemory, nullptr);
	}

	void CreateDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFlight, m_vkDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_vkDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(kMaxFramesInFlight);
		allocInfo.pSetLayouts = layouts.data();

		m_vkDescriptorSets.resize(kMaxFramesInFlight);
		if (vkAllocateDescriptorSets(m_vkDevice, &allocInfo, m_vkDescriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < kMaxFramesInFlight; i++)
		{
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer	= m_uniformBuffers[i];
			bufferInfo.offset	= 0;
			bufferInfo.range	= sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = m_textureImageView;
			imageInfo.sampler = m_textureSampler;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = m_vkDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;
			descriptorWrites[0].pImageInfo = nullptr; // Optional
			descriptorWrites[0].pTexelBufferView = nullptr; // Optional

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = m_vkDescriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(m_vkDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
		}
	}

	void CreateDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(kMaxFramesInFlight);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(kMaxFramesInFlight);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(kMaxFramesInFlight);

		if (vkCreateDescriptorPool(m_vkDevice, &poolInfo, nullptr, &m_vkDescriptorPool) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create descriptor pool!");
		}
	}

	void CreateUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		m_uniformBuffers.resize(kMaxFramesInFlight);
		m_uniformBuffersMemory.resize(kMaxFramesInFlight);
		m_uniformBuffersMapped.resize(kMaxFramesInFlight);

		for (size_t i = 0; i < kMaxFramesInFlight; i++) 
		{
			CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
			vkMapMemory(m_vkDevice, m_uniformBuffersMemory[i], 0, bufferSize, 0, &m_uniformBuffersMapped[i]);
		}
	}

	void CreateDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(m_vkDevice, &layoutInfo, nullptr, &m_vkDescriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create descriptor set layout!");
		}
	}

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(m_vkDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(m_vkDevice, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(m_vkDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to allocate buffer memory!");
		}

		vkBindBufferMemory(m_vkDevice, buffer, bufferMemory, 0);
	}

	void CreateVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();
	
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(m_vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, m_vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(m_vkDevice, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertexBuffer, m_vertexBufferMemory);

		CopyBuffer(stagingBuffer, m_vertexBuffer, bufferSize);

		vkDestroyBuffer(m_vkDevice, stagingBuffer, nullptr);
		vkFreeMemory(m_vkDevice, stagingBufferMemory, nullptr);
	}

	void CreateIndexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(m_vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, m_indices.data(), (size_t)bufferSize);
		vkUnmapMemory(m_vkDevice, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_indexBuffer, m_indexBufferMemory);

		CopyBuffer(stagingBuffer, m_indexBuffer, bufferSize);

		vkDestroyBuffer(m_vkDevice, stagingBuffer, nullptr);
		vkFreeMemory(m_vkDevice, stagingBufferMemory, nullptr);
	}

	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		EndSingleTimeCommands(commandBuffer);
	}

	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(m_vkPhysicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) 
		{
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type!");
	}

	void RecreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(m_window, &width, &height);
		while (width == 0 || height == 0) 
		{
			glfwGetFramebufferSize(m_window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(m_vkDevice);

		CleanupSwapChain();

		CreateSwapChain();
		CreateImageViews();
		CreateDepthResources();
		CreateFramebuffers();
	}

	void CleanupSwapChain()
	{
		vkDestroyImageView(m_vkDevice, m_depthImageView, nullptr);
		vkDestroyImage(m_vkDevice, m_depthImage, nullptr);
		vkFreeMemory(m_vkDevice, m_depthImageMemory, nullptr);

		for (auto framebuffer : m_vkSwapChainFramebuffers) 
		{
			vkDestroyFramebuffer(m_vkDevice, framebuffer, nullptr);
		}

		for (auto imageView : m_vkSwapChainImageViews) 
		{
			vkDestroyImageView(m_vkDevice, imageView, nullptr);
		}

		vkDestroySwapchainKHR(m_vkDevice, m_vkSwapChain, nullptr);
	}

	void DrawFrame()
	{
		vkWaitForFences(m_vkDevice, 1, &m_vkInFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(m_vkDevice, m_vkSwapChain, UINT64_MAX, m_vkImageAvailableSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) 
		{
			RecreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) 
		{
			throw std::runtime_error("Failed to acquire swap chain image!");
		}

		vkResetFences(m_vkDevice, 1, &m_vkInFlightFences[m_currentFrame]);
		vkResetCommandBuffer(m_vkCommandBuffers[m_currentFrame], 0);
		RecordCommandBuffer(m_vkCommandBuffers[m_currentFrame], imageIndex);

		UpdateUniformBuffer(m_currentFrame);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { m_vkImageAvailableSemaphores[m_currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_vkCommandBuffers[m_currentFrame];

		VkSemaphore signalSemaphores[] = { m_vkRenderFinishedSemaphores[m_currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(m_vkGraphicsQueue, 1, &submitInfo, m_vkInFlightFences[m_currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { m_vkSwapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr; // Optional

		result = vkQueuePresentKHR(m_vkPresentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResized)
		{
			m_framebufferResized = false;
			RecreateSwapChain();
		}
		else if (result != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to present swap chain image!");
		}

		m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
	}

	void UpdateUniformBuffer(uint32_t currentFrame)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		ubo.model	= glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view	= glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj	= glm::perspective(glm::radians(45.0f), m_vkSwapChainExtent.width / (float)m_vkSwapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		memcpy(m_uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void CreateSyncObjects()
	{
		m_vkImageAvailableSemaphores.resize(kMaxFramesInFlight);
		m_vkRenderFinishedSemaphores.resize(kMaxFramesInFlight);
		m_vkInFlightFences.resize(kMaxFramesInFlight);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (int i = 0; i < kMaxFramesInFlight; i++)
		{
			if (vkCreateSemaphore(m_vkDevice, &semaphoreInfo, nullptr, &m_vkImageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(m_vkDevice, &semaphoreInfo, nullptr, &m_vkRenderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(m_vkDevice, &fenceInfo, nullptr, &m_vkInFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create semaphores!");
			}
		}
	}

	void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) 
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0; // Optional
		beginInfo.pInheritanceInfo = nullptr; // Optional

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = m_vkRenderPass;
		renderPassInfo.framebuffer = m_vkSwapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = m_vkSwapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkGraphicsPipeline);

		VkBuffer vertexBuffers[] = { m_vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(m_vkSwapChainExtent.width);
		viewport.height = static_cast<float>(m_vkSwapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = m_vkSwapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkPipelineLayout, 0, 1, &m_vkDescriptorSets[m_currentFrame], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to record command buffer!");
		}
	}

	void CreateCommandBuffer()
	{
		m_vkCommandBuffers.resize(kMaxFramesInFlight);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType					= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool			= m_vkCommandPool;
		allocInfo.level					= VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount	= m_vkCommandBuffers.size();

		if (vkAllocateCommandBuffers(m_vkDevice, &allocInfo, m_vkCommandBuffers.data()) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to allocate command buffers!");
		}
	}

	void CreateCommandPool()
	{
		QueueFamilyIndices indices = FindQueueFamilies(m_vkPhysicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = indices.graphicsFamily.value();

		if (vkCreateCommandPool(m_vkDevice, &poolInfo, nullptr, &m_vkCommandPool) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create Vulkan command pool!");
		}
	}

	void CreateFramebuffers()
	{
		m_vkSwapChainFramebuffers.resize(m_vkSwapChainImageViews.size());

		for (size_t i = 0; i < m_vkSwapChainImageViews.size(); i++) 
		{
			std::array<VkImageView, 2> attachments =
			{
				m_vkSwapChainImageViews[i],
				m_depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = m_vkRenderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = m_vkSwapChainExtent.width;
			framebufferInfo.height = m_vkSwapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(m_vkDevice, &framebufferInfo, nullptr, &m_vkSwapChainFramebuffers[i]) != VK_SUCCESS) 
			{
				throw std::runtime_error("Failed to create framebuffer!");
			}
		}
	}

	void CreateRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format			= m_vkSwapChainImageFormat;
		colorAttachment.samples			= VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout		= VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment	= 0;
		colorAttachmentRef.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format			= FindDepthFormat();
		depthAttachment.samples			= VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp			= VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment	= 1;
		depthAttachmentRef.layout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount	= 1;
		subpass.pColorAttachments		= &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass		= VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass		= 0;
		dependency.srcStageMask		= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask	= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependency.dstStageMask		= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask	= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType			= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount	= static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments		= attachments.data();
		renderPassInfo.subpassCount		= 1;
		renderPassInfo.pSubpasses		= &subpass;
		renderPassInfo.dependencyCount	= 1;
		renderPassInfo.pDependencies	= &dependency;

		if (vkCreateRenderPass(m_vkDevice, &renderPassInfo, nullptr, &m_vkRenderPass) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create render pass!");
		}
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(m_vkDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create shader module!");
		}

		return shaderModule;
	}

	void CreateGraphicsPipeline()
	{
		auto vertShaderCode = ReadFile("shaders/vert.spv");
		auto fragShaderCode = ReadFile("shaders/frag.spv");
		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main"; // Entry point

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

		std::vector<VkDynamicState> dynamicStates = 
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		auto bindingDescription = Vertex::GetBindingDescription();
		auto attributeDescriptions = Vertex::GetAttributeDescriptions();
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
		vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {}; // Optional
		depthStencil.back = {}; // Optional

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

		// Multisampling
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional

		// Color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1; // Optional
		pipelineLayoutInfo.pSetLayouts = &m_vkDescriptorSetLayout; // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

		if (vkCreatePipelineLayout(m_vkDevice, &pipelineLayoutInfo, nullptr, &m_vkPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create Vulkan pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType					= VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount				= static_cast<uint32_t>(shaderStages.size());
		pipelineInfo.pStages				= shaderStages.data();
		pipelineInfo.pVertexInputState		= &vertexInputInfo;
		pipelineInfo.pInputAssemblyState	= &inputAssembly;
		pipelineInfo.pViewportState			= &viewportState;
		pipelineInfo.pRasterizationState	= &rasterizer;
		pipelineInfo.pMultisampleState		= &multisampling;
		pipelineInfo.pDepthStencilState		= &depthStencil; // Optional
		pipelineInfo.pColorBlendState		= &colorBlending;
		pipelineInfo.pDynamicState			= &dynamicState;
		pipelineInfo.layout					= m_vkPipelineLayout;
		pipelineInfo.renderPass				= m_vkRenderPass;
		pipelineInfo.subpass				= 0;
		pipelineInfo.basePipelineHandle		= VK_NULL_HANDLE; // Optional
		pipelineInfo.basePipelineIndex		= -1; // Optional

		if (vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_vkGraphicsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create Vulkan graphics pipeline!");
		}

		vkDestroyShaderModule(m_vkDevice, fragShaderModule, nullptr);
		vkDestroyShaderModule(m_vkDevice, vertShaderModule, nullptr);
	}

	void CreateImageViews()
	{
		m_vkSwapChainImageViews.resize(m_vkSwapChainImages.size());

		for (uint32_t i = 0; i < m_vkSwapChainImages.size(); i++) 
		{
			m_vkSwapChainImageViews[i] = CreateImageView(m_vkSwapChainImages[i], m_vkSwapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	void CreateVkSurface()
	{
		if (glfwCreateWindowSurface(m_vkInstance, m_window, nullptr, &m_vkSurface) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create Vulkan window surface!");
		}
	}

	void CreateLogicalDevice() 
	{
		QueueFamilyIndices indices = FindQueueFamilies(m_vkPhysicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		// Vulkan lets you assign priorities to queues to influence the scheduling of 
		// command buffer execution using floating point numbers between 0.0 and 1.0. 
		// This is required even if there is only a single queue:
		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) 
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(g_physDeviceExtensions.size()); 
		createInfo.ppEnabledExtensionNames = g_physDeviceExtensions.data();

		if (kEnableVkValidationLayers) 
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(g_vkValidationLayers.size());
			createInfo.ppEnabledLayerNames = g_vkValidationLayers.data();
		}
		else 
		{
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(m_vkPhysicalDevice, &createInfo, nullptr, &m_vkDevice) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create Vulkan logical device!");
		}

		// Get handle to graphics queue
		vkGetDeviceQueue(m_vkDevice, indices.graphicsFamily.value(), 0, &m_vkGraphicsQueue);
		vkGetDeviceQueue(m_vkDevice, indices.presentFamily.value(), 0, &m_vkPresentQueue);
	}

	// Picks the first graphics card
	void PickPhysicalDevice() 
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, nullptr);

		if (deviceCount == 0) 
		{
			throw std::runtime_error("Failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, devices.data());

		for (const auto& device : devices) 
		{
			if (IsPhysicalDeviceSuitable(device)) 
			{
				m_vkPhysicalDevice = device;
				break;
			}
		}

		if (m_vkPhysicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("Failed to find a suitable GPU!");
		}
	}

	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) 
	{
		QueueFamilyIndices indices{};
		
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		for (uint32_t i = 0; i < queueFamilyCount; i++)
		{
			if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_vkSurface, &presentSupport);

			if (presentSupport) 
			{
				indices.presentFamily = i;
			}

			if (indices.IsComplete()) break;
		}

		return indices;
	}

	bool IsPhysicalDeviceSuitable(VkPhysicalDevice device) 
	{
		QueueFamilyIndices indices = FindQueueFamilies(device);

		bool extensionsSupported = CheckPhysicalDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) 
		{
			SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return 
			indices.IsComplete() && 
			extensionsSupported && 
			swapChainAdequate &&
			supportedFeatures.samplerAnisotropy;
	}

	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) 
	{
		for (const auto& availableFormat : availableFormats) 
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
				availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) 
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) 
	{
		for (const auto& availablePresentMode : availablePresentModes) 
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) 
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) 
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) 
		{
			return capabilities.currentExtent;
		}
		else 
		{
			int width, height;
			glfwGetFramebufferSize(m_window, &width, &height);

			VkExtent2D actualExtent = 
			{
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	void CreateSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(m_vkPhysicalDevice);
		VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && 
			imageCount > swapChainSupport.capabilities.maxImageCount) 
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType			= VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface			= m_vkSurface;
		createInfo.minImageCount	= imageCount;
		createInfo.imageFormat		= surfaceFormat.format;
		createInfo.imageColorSpace	= surfaceFormat.colorSpace;
		createInfo.imageExtent		= extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage		= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = FindQueueFamilies(m_vkPhysicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) 
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else 
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}

		createInfo.preTransform		= swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha	= VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode		= presentMode;
		createInfo.clipped			= VK_TRUE;
		createInfo.oldSwapchain		= VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(m_vkDevice, &createInfo, nullptr, &m_vkSwapChain) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create Vulkan swapchain!");
		}

		vkGetSwapchainImagesKHR(m_vkDevice, m_vkSwapChain, &imageCount, nullptr);
		m_vkSwapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(m_vkDevice, m_vkSwapChain, &imageCount, m_vkSwapChainImages.data());

		m_vkSwapChainImageFormat = surfaceFormat.format;
		m_vkSwapChainExtent = extent;
	}

	bool CheckPhysicalDeviceExtensionSupport(VkPhysicalDevice device) 
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(g_physDeviceExtensions.begin(), g_physDeviceExtensions.end());

		for (const auto& extension : availableExtensions) 
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	void CreateVkInstance()
	{
		if (kEnableVkValidationLayers && !CheckVkValidationLayerSupport())
		{
			throw std::runtime_error("Validation layers requested, but not available!");
		}

		// Technically optional, but may provide useful info to the driver
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		// Gather available extensions and print
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

		std::cout << "Available Vulkan extensions:\n";
		for (const auto& extension : extensions) 
		{
			std::cout << '\t' << extension.extensionName << '\n';
		}
		std::cout << std::endl;

		// Tell vk what extensions to use and how many
		auto glfwExtensions = GetRequiredVkExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
		createInfo.ppEnabledExtensionNames = glfwExtensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (kEnableVkValidationLayers) 
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(g_vkValidationLayers.size());
			createInfo.ppEnabledLayerNames = g_vkValidationLayers.data();

			PopulateVkDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else 
		{
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &m_vkInstance) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to create Vulkan instance!");
		}
	}

	bool CheckVkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		// Check if all g_vkValidationLayers exist
		for (const char* layerName : g_vkValidationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) 
			{
				if (strcmp(layerName, layerProperties.layerName) == 0) 
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound) 
			{
				return false;
			}
		}

		return true;
	}

	std::vector<const char*> GetRequiredVkExtensions() 
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (kEnableVkValidationLayers) 
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	void PopulateVkDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) 
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = DebugCallback;
	}

	void SetupVkDebugMessenger() 
	{
		if (!kEnableVkValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		PopulateVkDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(m_vkInstance, &createInfo, nullptr, &m_vkDebugMessenger) != VK_SUCCESS) 
		{
			throw std::runtime_error("Failed to set up Vulkan debug messenger!");
		}
	}

	void MainLoop()
	{
		while (!glfwWindowShouldClose(m_window)) 
		{
			glfwPollEvents();
			DrawFrame();
		}

		vkDeviceWaitIdle(m_vkDevice);
	}

	SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) const
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_vkSurface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_vkSurface, &formatCount, nullptr);

		if (formatCount != 0) 
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_vkSurface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_vkSurface, &presentModeCount, nullptr);

		if (presentModeCount != 0) 
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_vkSurface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	void Cleanup()
	{
		CleanupSwapChain();

		vkDestroySampler(m_vkDevice, m_textureSampler, nullptr);
		vkDestroyImageView(m_vkDevice, m_textureImageView, nullptr);

		vkDestroyImage(m_vkDevice, m_textureImage, nullptr);
		vkFreeMemory(m_vkDevice, m_textureImageMemory, nullptr);

		vkDestroyDescriptorPool(m_vkDevice, m_vkDescriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayout, nullptr);

		for (size_t i = 0; i < kMaxFramesInFlight; i++)
		{
			vkDestroyBuffer(m_vkDevice, m_uniformBuffers[i], nullptr);
			vkFreeMemory(m_vkDevice, m_uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayout, nullptr);

		vkDestroyBuffer(m_vkDevice, m_indexBuffer, nullptr);
		vkFreeMemory(m_vkDevice, m_indexBufferMemory, nullptr);

		vkDestroyBuffer(m_vkDevice, m_vertexBuffer, nullptr);
		vkFreeMemory(m_vkDevice, m_vertexBufferMemory, nullptr);

		vkDestroyPipeline(m_vkDevice, m_vkGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(m_vkDevice, m_vkPipelineLayout, nullptr);
		vkDestroyRenderPass(m_vkDevice, m_vkRenderPass, nullptr);

		for (size_t i = 0; i < kMaxFramesInFlight; i++) 
		{
			vkDestroySemaphore(m_vkDevice, m_vkImageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(m_vkDevice, m_vkRenderFinishedSemaphores[i], nullptr);
			vkDestroyFence(m_vkDevice, m_vkInFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(m_vkDevice, m_vkCommandPool, nullptr);
		vkDestroyDevice(m_vkDevice, nullptr);

		if (kEnableVkValidationLayers) 
		{
			DestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(m_vkInstance, m_vkSurface, nullptr);
		vkDestroyInstance(m_vkInstance, nullptr);

		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

public:
	bool							m_framebufferResized = false;

private:
	GLFWwindow*						m_window;
	VkSurfaceKHR					m_vkSurface;
	VkInstance						m_vkInstance;
	VkDebugUtilsMessengerEXT		m_vkDebugMessenger;
	VkPhysicalDevice				m_vkPhysicalDevice = VK_NULL_HANDLE;
	VkDevice						m_vkDevice;
	VkQueue							m_vkGraphicsQueue;
	VkQueue							m_vkPresentQueue;
	VkSwapchainKHR					m_vkSwapChain;
	std::vector<VkImage>			m_vkSwapChainImages;
	std::vector<VkImageView>		m_vkSwapChainImageViews;
	std::vector<VkFramebuffer>		m_vkSwapChainFramebuffers;
	VkFormat						m_vkSwapChainImageFormat;
	VkExtent2D						m_vkSwapChainExtent;
	VkRenderPass					m_vkRenderPass;
	VkDescriptorPool				m_vkDescriptorPool;
	std::vector<VkDescriptorSet>	m_vkDescriptorSets;
	VkDescriptorSetLayout			m_vkDescriptorSetLayout;
	VkPipelineLayout				m_vkPipelineLayout;
	VkPipeline						m_vkGraphicsPipeline;
	VkCommandPool					m_vkCommandPool;
	std::vector<VkCommandBuffer>	m_vkCommandBuffers;
	std::vector<VkSemaphore> 		m_vkImageAvailableSemaphores;
	std::vector<VkSemaphore> 		m_vkRenderFinishedSemaphores;
	std::vector<VkFence> 			m_vkInFlightFences;

	uint32_t						m_currentFrame;

	// Geometry
	VkBuffer						m_vertexBuffer;
	VkDeviceMemory					m_vertexBufferMemory;
	VkBuffer						m_indexBuffer;
	VkDeviceMemory					m_indexBufferMemory;
	std::vector<Vertex>				m_vertices;
	std::vector<uint32_t>			m_indices;

	// Texture
	VkImage							m_textureImage;
	VkDeviceMemory					m_textureImageMemory;
	VkImageView						m_textureImageView;
	VkSampler						m_textureSampler;

	// UBO
	std::vector<VkBuffer>			m_uniformBuffers;
	std::vector<VkDeviceMemory>		m_uniformBuffersMemory;
	std::vector<void*>				m_uniformBuffersMapped;

	// Depth testing
	VkImage							m_depthImage;
	VkDeviceMemory					m_depthImageMemory;
	VkImageView						m_depthImageView;
};

static void FramebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
	app->m_framebufferResized = true;
}

int main()
{
	Application app{};

	try 
	{
		app.Run();
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}